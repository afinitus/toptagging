def ODESampler(self, cond, model, data_shape=None, init_noise=None, jet=None, mask=None, atol=1e-5, eps=1e-5):
    from scipy import integrate
    batch_size = cond.shape[0]

    if init_noise is None:
        init_noise = self.prior_sde(data_shape)
    shape = init_noise.shape

    @tf.function
    def score_eval_wrapper(sample, time_steps, cond, jet=None, mask=None):
        sample = tf.cast(tf.reshape(sample, shape), tf.float32)
        time_steps = tf.reshape(time_steps, (sample.shape[0], 1))
        time_steps = self.inv_logsnr_schedule_cosine(2 * time_steps,
                                                     logsnr_min=self.minlogsnr,
                                                     logsnr_max=self.maxlogsnr)
        logsnr_steps, alpha, sigma = self.get_logsnr_alpha_sigma(time_steps, shape=(-1, 1))
    
        if mask is not None:
            sample = self.Featurizer(sample) * mask
        
        pred = self.eval_model(model, sample, time_steps, jet, mask)
        drift = -sigma * alpha * pred
        return tf.reshape(drift, [-1])

    def ode_func(t, x):
        time_steps = np.ones((batch_size, 1), dtype=np.float32) * t
        drift = score_eval_wrapper(x, time_steps, cond, jet, mask).numpy()
        return drift

    integration_bounds = (self.minlogsnr/2.0 + eps, self.maxlogsnr/2.0 - eps)
    res = integrate.solve_ivp(
        ode_func, 
        integration_bounds, 
        tf.reshape(init_noise, [-1]).numpy(),
        rtol=atol, 
        atol=atol, 
        method='RK45'
    )
    
    sample = tf.reshape(res.y[:, -1], shape)
    return sample
