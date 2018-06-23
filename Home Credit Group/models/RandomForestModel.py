class RandomForestModel(Model):

    def train(self, train_dataset, eval_dataset=None, num_round=None, verbose=False):
        print("Start training model")
        raise NotImplementedError()

    def val(self, eval_dataset, y_true, threshold=0.5):
        raise NotImplementedError()

    def infer(self, test_dataset, threshold=0.5, output_as='logits', to_csv=False):
        raise NotImplementedError()