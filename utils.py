def train_log(epoch, batch_step, total_step, loss):
    print('Epoch {}, {} / {}, loss: {}'.format(epoch, batch_step, total_step,
                                               loss.item()))
