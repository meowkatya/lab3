# Практическая работа 3
Выполнила: Кузнецова Е.А. группа R3235.
Преподаватель: Каканов М.А., Евстафьев О.А.

# Цель работы:
Познакомиться с различными архитектурами сверхточных нейронных сетей и их обучением на GPU.

# Ход работы:
```python
# Выбираем предобученную модель
model = torchvision.models.googlenet(pretrained=True)
model
# Передаем модель в функцию градиента
set_parameter_requires_grad(model)
# Меняем последний слой модели
model.fc = nn.Linear(1024, 5)
# Выводим веса, которые будут обучаться
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
# Отправляем модель на GPU
model.to(device) 
optimizer = torch.optim.Adam(params = model.parameters()) 
# Запускаем обучение
loss_track, accuracy_track = trainval(model, loaders, optimizer, epochs=10)
plt.plot(accuracy_track['training'], label='training')
plt.plot(accuracy_track['validation'], label='validation')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid()
plt.legend()
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

def predict_image(img, model):
    # Преобразование to a batch of 1
    xb = img.unsqueeze(0).to(device)
    # Получение прогнозов от модели
    yb = model(xb)
    # Выбираем индекс с наибольшей вероятностью
    _, preds  = torch.max(yb, dim=1)
    # Получение метки класса
    return dataset.classes[preds[0].item()]

for i in range(1,10):
  img, label = val_set[i]
  plt.imshow(img.clip(0,1).permute(1, 2, 0))
  plt.axis('off')
  plt.title('Label: {}, Predicted: {}'.format(dataset.classes[label],predict_image(img, model)))
  plt.show()
# print('Label:', dataset.classes[label], ',Predicted:', predict_image(img, ______))
# Сохраняем веса модели
weights_fname = '/content/drive/MyDrive/data/flowers.pth'
torch.save(model.state_dict(), weights_fname)
```

# Вывод:
Познакомились с различными архитектурами сверхточных нейронных сетей и их обучением на GPU с помощью Google Colab - Colaboratory.
