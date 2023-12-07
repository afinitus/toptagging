def get_likelihood(self,part,jet,mask):
    start = time.time()
    ll_jet, final_noise_jet = self.Likelihood(jet,self.ema_jet)
    ll_part, final_noise_part = self.Likelihood(part,self.ema_part,jet=jet,mask=mask)
    return ll_part, ll_jet, final_noise_part, final_noise_jet

#removed second input param jet_info
def generate(self,num_jets):
    start = time.time()
    jet_info, init_noise_jets = self.ODESampler(num_jets,self.ema_jet,
                                data_shape=[num_jets, self.num_jet])
    
    end = time.time()
    print("Time for sampling {} events is {} seconds".format(num_jets,end - start))

    nparts = np.expand_dims(np.clip(utils.revert_npart(jet_info[:,-1],self.max_part),
                                    0,self.max_part),-1)
    

    mask = np.expand_dims(
        np.tile(np.arange(self.max_part),(nparts.shape[0],1)) < np.tile(nparts,(1,self.max_part)),-1)

    start = time.time()
    parts, init_noise_parts = self.ODESampler(tf.convert_to_tensor(num_jets,dtype=tf.int32),
                            self.ema_part,
                            data_shape=[num_jets, self.max_part, self.num_feat],
                            jet=tf.convert_to_tensor(jet_info),
                            mask=tf.convert_to_tensor(mask, dtype=tf.float32))

    # parts = self.DDPMSampler(tf.convert_to_tensor(cond,dtype=tf.float32),
    #                         self.ema_part,
    #                         data_shape=[self.max_part,self.num_feat],
    #                         jet=tf.convert_to_tensor(jet_info, dtype=tf.float32),
    #                         mask=tf.convert_to_tensor(mask, dtype=tf.float32)).numpy()

    # parts = np.ones(shape=(cond.shape[0],self.max_part,3))
    end = time.time()
    print("Time for sampling {} events is {} seconds".format(num_jets,end - start))
    return parts*mask,jet_info, init_noise_parts, init_noise_jets, mask

def ODESampler(self, num_jets, model, data_shape=None, init_noise=None, jet=None, mask=None, atol=1e-2, eps=1e-5):
    from scipy import integrate
    batch_size = num_jets

    if init_noise is None:
        init_noise = self.prior_sde(data_shape)
    shape = init_noise.shape
    #print(init_noise)
    #shape defining
    if mask is None:
        const_shape = (-1,1)
    else:
        const_shape = self.shape
    
    @tf.function
    def score_eval_wrapper(sample, time_steps, jet=None, mask=None):
        sample = tf.cast(tf.reshape(sample, shape), tf.float32)
        #print(sample, "sample shape")
        time_steps = tf.reshape(time_steps, (sample.shape[0], 1))
        time_steps = self.inv_logsnr_schedule_cosine(2 * time_steps,
                                                        logsnr_min=self.minlogsnr,
                                                        logsnr_max=self.maxlogsnr)
        
        logsnr_steps, alpha, sigma = self.get_logsnr_alpha_sigma(time_steps, shape=const_shape)


        pred = self.eval_model(model, sample, time_steps, jet, mask)
        drift = -sigma * alpha * pred
        return tf.reshape(drift, [-1])

    def ode_func(t, x):
        time_steps = np.ones((batch_size, 1), dtype=np.float32) * t
        drift = score_eval_wrapper(x, time_steps, jet, mask).numpy()
        return drift

    integration_bounds = (self.minlogsnr/2.0 + eps, self.maxlogsnr/2.0 - eps)
    res = integrate.solve_ivp(
        ode_func, 
        integration_bounds, 
        tf.reshape(init_noise, [-1]).numpy(),
        rtol=atol, 
        atol=atol, 
        method='RK23'
    )

    sample = tf.reshape(res.y[:, -1], shape)
    return sample, init_noise


def Likelihood(self,
                sample,
                model,
                jet=None,
                mask=None,
                atol=1e-2,
                eps=1e-5,
                exact = True,
):

    from scipy import integrate

    gc.collect()
    batch_size = sample.shape[0]        
    shape = sample.shape

    
    if mask is None:
        N = np.prod(shape[1:])
        const_shape = (-1,1)
    else:
        N = np.sum(self.num_feat*mask,(1,2))
        const_shape = self.shape

        
    def prior_likelihood(z):
        """The likelihood of a Gaussian distribution with mean zero and 
        standard deviation sigma."""
        shape = z.shape            
        return -N / 2. * np.log(2*np.pi) - np.sum(z.reshape((shape[0],-1))**2, -1) / 2. 

    
    @tf.function
    def divergence_eval_wrapper(sample, time_steps,
                                jet=None,mask=None):
        
        sample = tf.cast(tf.reshape(sample,shape),tf.float32)
        time_steps = tf.reshape(time_steps,(sample.shape[0], 1))
        time_steps = self.inv_logsnr_schedule_cosine(2*time_steps,
                                                        logsnr_min=self.minlogsnr,
                                                        logsnr_max=self.maxlogsnr
                                                        )
        
        logsnr_steps, alpha, sigma = self.get_logsnr_alpha_sigma(time_steps,shape=const_shape)
        epsilons = tfp.random.rademacher(sample.shape,dtype=tf.float32)
        #epsilons = tf.random.normal(sample.shape,dtype=tf.float32)            
        if mask is not None:
            sample*=mask
            epsilons*=mask

        if exact:
            # Exact trace estimation
            fn = lambda x: -sigma*alpha*self.eval_model(model, x,
                        (logsnr_steps-self.minlogsnr)/(self.maxlogsnr -self.minlogsnr),jet,mask)
        

            pred, diag_jac = diag_jacobian(
            xs=sample, fn=fn, sample_shape=[batch_size])

            if isinstance(pred, list):
                pred = pred[0]
                if isinstance(diag_jac, list):
                    diag_jac = diag_jac[0]
        
            return tf.reshape(pred,[-1]), - tf.reduce_sum(tf.reshape(diag_jac,(batch_size,-1)), -1)
        else:
            
            with tf.GradientTape(persistent=False,
                                    watch_accessed_variables=False) as tape:
                tape.watch(sample)
                pred = self.eval_model(model,sample,
                                        (logsnr_steps-self.minlogsnr)/(self.maxlogsnr -self.minlogsnr),
                                        jet,mask)                
                drift = -sigma*alpha*pred
            
            jvp = tf.cast(tape.gradient(drift, sample,epsilons),tf.float32)            
            return  tf.reshape(drift,[-1]), - tf.reduce_sum(tf.reshape(jvp*epsilons,(batch_size,-1)), -1)



    def ode_func(t, x):        
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((batch_size,)) * t    
        sample = x[:-batch_size]
        logp = x[-batch_size:]
        sample_grad, logp_grad = divergence_eval_wrapper(sample, time_steps,jet,mask)
        return np.concatenate([sample_grad, logp_grad], axis=0)

    init_x = np.concatenate([sample.reshape([-1]),np.zeros((batch_size,))],0)
    res = integrate.solve_ivp(
        ode_func,
        #(eps,1.0-eps),
        (self.maxlogsnr/2.0 - eps,self.minlogsnr/2.0+eps),
        init_x,
        #max_step=5e-3,
        rtol=atol, atol=atol, method='RK23')

    zp = res.y[:, -1]
    z = zp[:-batch_size].reshape(shape)
    if mask is not None:
        z *= mask
        
    delta_logp = zp[-batch_size:].reshape(batch_size)
    prior_logp = prior_likelihood(z)
    return (prior_logp - delta_logp), z
