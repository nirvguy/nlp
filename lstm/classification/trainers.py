import torchtrainer

class SupervisedValidator(torchtrainer.SupervisedValidator):
    def validate_batch(self, lines, lengths, category):
        output = self.model(lines, lengths)

        for meter in self.meters.values():
            meter.measure(output.data, category.data)

class SupervisedTrainer(torchtrainer.SupervisedTrainer):
    def create_validator(self):
        return SupervisedValidator(self.model, self.val_meters)

    def update_batch(self, lines, lengths, category):
        self.optimizer.zero_grad()
        output = self.model(lines, lengths)
        loss = self.criterion(output, category)
        loss.backward()
        self.optimizer.step()

        for meter in self.train_meters.values():
            meter.measure(output.data, category.data)
