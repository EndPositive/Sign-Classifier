from models import Classifier
import pickle

model_path = './models/model7.hd5'

# load dataset
with open('./datasets/80k_no_priorityroad_grey.pickle', 'rb') as f:
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = pickle.load(f)


if __name__ == '__main__':
    # make model
    model = Classifier()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train model
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))

    # test model
    model.evaluate(x_test, y_test)

    # save model
    model.save(model_path)
    print('MODEL SAVED')
