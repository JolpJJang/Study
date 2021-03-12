from keras.applications.resnet50 import ResNet50

model = ResNet50(weights=None)
model.summary()