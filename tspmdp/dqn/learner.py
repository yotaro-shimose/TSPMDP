class Learner:
    def start(self):
        self._initialize()
        for epoch in self.n_epochs:
            metrics = self._train()
            self.logger.log_metrics(metrics)
            self._per_step_process()

    def _initialize(self):
        self._build()

    def _build(self):
        # Build networkinstance lazily here.
        # Build network's weights here
        raise NotImplementedError

    def _train(self):
        batch = self.server.sample(self.batch_size)
        args = self.create_args(batch)
        raw_metrics = self._train_on_batch(args)
        return self.create_metrics(raw_metrics)

    def _train_on_batch(self, **kwargs):
        raise NotImplementedError

    def _synchronize(self):
        self.network_target.set_weights(self.network.get_weights())

    def _upload(self):
        self.server.upload(self.network.get_weights())

    def _per_step_process(self):
        # Synchronize weights

        # Upload weights

        raise NotImplementedError
