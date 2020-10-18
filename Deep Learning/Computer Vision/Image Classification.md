

[TOC]



# CNN matome



## Q&A

- CNNは層を深くすればするほどいいの？
- 自動でハイパーパラメータ探索できないの？
  - Keras-tuner 

- 転移学習は異なるドメインでなぜ有効なの？
- CNN は何を見ているの？



## 歴史



### 概観



**深く・深く・・の時代**

〜2015までは、層を深くすればするほど精度が上がる、もっと深くするにはどうすればいいだろう？の時代。

層を深くしたくても、勾配消失や過学習の問題があり、それをどう工夫して解決するかがポイント



層を深くしたい or モデルを軽量化したい

- Auciliary Loss（中間層から直接勾配を流す）

- Skip connection / residual モジュール（勾配を直接下層に流す）

↑

パラメータ数を削減する or 扱う画像のサイズを小さくする

- 1x1 conv, Bottleneck 

- 畳み込みの分解
  - Grouped Conv
  - Depthwise Separable Conv



**深さの限界に達する**

深くしていけば行くほど良いかと思いきや、深くしていっても限界がある。2倍深くすれば2倍精度上がるかといえばそうではない。ResNet1000とReNet101はほぼ同じ精度。

↓

広さの探索、解像度の探索（WideResNet や EddicientNet）、その他オリジナル構成（DenseNet, SENetなど）



**性能を落とさずにモデルサイズを小さくできないか**

- SqueezeNet

  - Fire Module : 1x1 conv でチャネル方向の次元削減 + 枝分かれ

  - ```python
    def fire_module(x, squeeze=16, expand=64):
        x = Convolution2D(squeeze, (1, 1), padding='same')(x)
        x = Activation('relu')(x)
    
        left = Convolution2D(expand, (1, 1), padding='same')(x)
        left = Activation('relu')(left)
    
        right = Convolution2D(expand, (3, 3), padding='same')(x)
        right = Activation('relu')(right)
    
        x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
        return x
    ```

<img src="/Users/kikagaku/Desktop/conv_cheetsheet.png" alt="conv_cheetsheet" style="zoom:50%;" />

https://medium.com/@yu4u/why-mobilenet-and-its-variants-e-g-shufflenet-are-fast-1c7048b9618d





**Neural Architecture Search**







### 年表

| year | name         | Depth           | point                                                        | Size |
| ---- | ------------ | --------------- | ------------------------------------------------------------ | ---- |
| 2012 | AlexNet      | 5               | ・ILSVRC(2012) で従来のハンドクラフトアプローチに大差つけて優勝。<br>・ksize=7 や 11 とか stride=4 とか。<br> ・Dropout |      |
| 2013 | ZFNet        | 5               | ・ILSVRC(2013) 優勝モデル。AlexNetを微修正                   |      |
| 2014 | GoogLeNet    | 22              | ・ILSVRC(2014) 優勝モデル。<br>・Inception モジュール。Network In Network<br>・Global Average Pooling <br>・1x1 conv<br>・Auxiliary Loss |      |
| 2014 | VGGNet       | 11, 13, 16, 19  | ・ILSVRC(2014) 2位モデル。<br>・3x3 conv のみ使用<br>・規則性のある構造（pool後チャネル数2倍）<br>・Glorot の初期化 |      |
| 2015 | ResNet       | 18, 34, 50, 152 | ・ILSVRC(2015) 優勝モデル。<br>・skip connection / residual モジュール / bottleneck モジュール<br>・Batch Normalization<br>・He の初期化 |      |
| 2016 | DenseNet     |                 | ・DenseBlock <br>・Growth rate（成長率）                     |      |
| 2017 | SENet        |                 | ・ILSVRC(2017) 優勝モデル。<br/>・Attention を活用した SE Block |      |
| 2017 | NASNet       |                 | ・Neural Architecture Search                                 |      |
| 2019 | EfficientNet |                 | ・NASによるベースモデル→Wide, Depth, Resolution を同時に考慮したスケーリングで当時のSOTA<br>・従来よりもパラメータ数減。結構シンプル<br>・物体検出など他タスクへのバックボーンとしての使用も効果的 |      |
| 2020 | ResNeSt      |                 | ・grouped conv と SE block を組み合わせた split attention を、cardinal の数だけ分割して更に複雑化した ResNeSt Block の導入で SOTA <br>・物体検出など他タスクへのバックボーンとしての使用も効果的 |      |
| 2020 | SAN          |                 |                                                              |      |



### シリーズ



#### Inception 系

- Inception v1 ( GoogLeNet )
- Inception v2
- Inception v3
- Inception v4
- Xception
- Batch norm の導入や、 nx1. 1xn の畳み込み分解



#### ResNet 系

- ResNet
- WideResNet

- ResNeXt
- ResNeSt



#### MobileNet 系

- MobileNet v1
- MobileNet v2



#### NASNet 系

- NasNet
- MNasNet 



## 主要技術



### Inception モジュール（2014）



### Global Average Pooling（2014）

- Flatten して全結合層にわたすとパラメータ数が非常に多くなる＋過学習を起こしやすくなる課題を緩和
- 現在は Flatten の代わりに GAP 入れるのがベストプラクティス



### Auxiliary Loss

- ネットワークの途中から分岐させたサブネットワークでもクラス分類を行い、auxiliary lossを追加
- 誤差逆伝播の際に、ネットワークの中間層に、直接誤差を伝播させることができる→勾配消失しにくい→層を深くしやすい＋ネットワークの正則化の効果もある。



### Skip Connection / Residual モジュール

- conv - bn - relu - conv - bn - relu （ ResNet 論文）
- この並び順にも議論がある。 
  - bn - relu - conv - bn - relu - conv - add
  - ショートカットの後にReLUを通さず、勾配がそのまま入力に近い層に伝わるのが良い？
  - bn 入れる順番や、dropout 入れるか？なども。



### 1 x 1 conv / Bottleneck モジュール

- 1 x 1 conv を挟むことでパラメータ数削減！



```python
def _bn_relu_conv(input, filters=3, kernel_size=3, strides=(1, 1), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1.e-4)):
    x = BatchNormalization()(input)
    x = Activation("relu")(x)
    out = Conv2D(filters=filters, kernel_size=kernel_size,
                    strides=strides, padding=padding,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer)
    return out

def basic_block(input, filters, init_strides=(1, 1)):
    conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3), strides=init_strides)(input)
    residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
    return add[input, residual]

def bottleneck(input, filters, init_strides=(1, 1)):
    conv_1_1 = _bn_relu_conv(filtesr=filters, kernel_size=(1, 1), strides=init_strides)(input)
    conv_3_3 = _bn_relu_conv(filtesr=filters, kernel_size=(3, 3)(conv_1_1)
    residual = _bn_relu_conv(filters=filters*4, kernel_size=(1, 1))(conv_3_3)
    return add[input, residual]
```





### Grouped Convolution

- ResNeXt の ResNeXt モジュール（←これは分割した処理結果を足し合わせるのでちょっと例外。一般的には結合）
- 入力特徴マップをチャネル方向で g 分割し、それぞれ独立に畳み込みを行い、結合する処理

<img src="/Users/kikagaku/Library/Application Support/typora-user-images/スクリーンショット 2020-09-02 11.47.59.png" alt="スクリーンショット 2020-09-02 11.47.59" style="zoom:50%;" />

```python
def __grouped_conv_block(input, grouped_channels, cardinality, strides):
    init = input 
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    group_list = []

    if cardinality==1:
        x = Conv2D(grouped_channels, (3, 3), padding="same", strides=(strides, strides))(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation("relu")(x)
        return x 
    
    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels: (c+1)*grouped_channels])(input)
        x = Conv2D(grouped_channels, (3, 3), padding="same")(x)
        group_list.append(x)
    
    group_merge = concatenate(group_list, axis=channel_axis)
    x = BatchNormalization(axis=channel_axis)(group_merge)
    x = Activation("relu")(x)

    return x
```





### Depthwise Separable Convolution

- MobileNet, Xception





### SEブロック（Squeeze and Excitation）

- 特徴マップのチャネル方向に重み付けをして、情報価値の高いチャネルの情報を強調する。

```python
# ration : 圧縮率
def se_block(input_block, ch, ratio=16):
  # squeeze：各チャネルを1つの値に集約する
  x = GlovalAveragePooling2D()(input_block)
  
  # excitation
  x = Dense(channel // ratio, activation='relu')(x) # チャネルを圧縮して
  x = Dense(channel, activation='sigmoid')(x) # 元のチャネル数に戻す。この際にsigmoidで非線形変換するので、0~1に圧縮する。すなわち、0 ~ 1 のレンジでの重み付けの情報となる。
  
  return multiply()([input_block, x])
```

### <img src="/Users/kikagaku/Library/Application Support/typora-user-images/スクリーンショット 2020-09-01 23.41.47.png" alt="スクリーンショット 2020-09-01 23.41.47" style="zoom:50%;" />

https://qiita.com/Q_ys/items/2054a8a724d22bd10aff



- 【要確認】convよりパラメータ数減少するはず？



## 学習のテクニック



### 疑似ラベル（pseudo-labeling）

- Noisy Student
  - ラベル付き画像で teacher モデルを学習
  - 学習済み teacher モデルを使って、ラベルなし画像に対して疑似ラベルを付与
  - ラベル付き画像と疑似ラベルつき画像をあわせて学習用データセットとし、teacher モデルより大きな student モデルを学習。その際以下でノイズをかける。
    - RandAugment
    - Dropout：ランダムにWide 幅を狭くする
    - Stochastic Depth ： Dropout を層に適用したもの。ランダムに Depth 幅を狭くする





## 実装して理解を深める



### Residual module / Bottleneck module

```python
def _bn_relu_conv(input, filters=3, kernel_size=3, strides=(1, 1), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1.e-4)):
    x = BatchNormalization()(input)
    x = Activation("relu")(x)
    out = Conv2D(filters=filters, kernel_size=kernel_size,
                    strides=strides, padding=padding,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer)
    return out

def basic_block(input, filters, init_strides=(1, 1)):
    conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3), strides=init_strides)(input)
    residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
    return add[input, residual]

def bottleneck(input, filters, init_strides=(1, 1)):
    conv_1_1 = _bn_relu_conv(filtesr=filters, kernel_size=(1, 1), strides=init_strides)(input)
    conv_3_3 = _bn_relu_conv(filtesr=filters, kernel_size=(3, 3)(conv_1_1)
    residual = _bn_relu_conv(filters=filters*4, kernel_size=(1, 1))(conv_3_3)
    return add[input, residual]
```



```python
def residual_block(input, filters):
  x = BatchNormalization()(input)
  x = Activation('relu')(x)
  x = Conv2D(filters=filters, kernel_size=3, stride=(1, 1))(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(filters=filters, kernel_size=3, stride=(1, 1))(x)

```





### ResNeXt module

```python
def __grouped_conv_block(input, grouped_channels, cardinality, strides):
    init = input 
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    group_list = []

    if cardinality==1:
        x = Conv2D(grouped_channels, (3, 3), padding="same", strides=(strides, strides))(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation("relu")(x)
        return x 
    
    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels: (c+1)*grouped_channels])(input)
        x = Conv2D(grouped_channels, (3, 3), padding="same")(x)
        group_list.append(x)
    
    group_merge = concatenate(group_list, axis=channel_axis)
    x = BatchNormalization(axis=channel_axis)(group_merge)
    x = Activation("relu")(x)

    return x

def resnext_module(input, filters=64, cardinality=8, stride=1):
    grouped_channels = int(filters / cardinality)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # conv -> bn -> relu
    x = Conv2D(filters, (1, 1), padding="same")(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation("relu")(x)

    # grouped_conv -> conv -> bn
    x = __grouped_conv_block(x, grouped_channels, cardinality, strides)
    x = Conv2D(filters*2, (1, 1), padding="same")(x)
    x = BatchNormalization(axis=channel_axis)(x)

    # add
    x = add([input, x])
    x = Activation("relu")(x)

    return x
```



### Squeeze and Excitation Block



```python
def se_block(input, output_dim, ratio):
    # squeeze
    x = GlobalAveragePooling2D()(input)

    # excitation
    x = Dense(units=output_dim // ratio, activation="relu")(x)
    x = Dense(units=output_dim, activation="sigmoid")(x)
    x = Reshape((1, 1, output_dim))(x)

    out = multiply([input, x])

    return out 
```



## TensorFlow 2.x 



### モデルの保存と読み込み

saved_model 

- `model.save` -> `tf.keras.load_model` 
  - HFD5
- `tf.saved_model.save` -> `tf.saved_model.load`



### TFDataset, TFRecord





### 分散学習

model 構築〜コンパイルを、`with mirrored_strategt.score():` でラップする。

```python
mirrored_strategy = tf.distribute.MirroredStrategy()

def get_model():
  with mirrored_strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model
```





### Data Augmentation

https://sayak.dev/tf.keras/data_augmentation/image/2020/05/10/augmemtation-recipes.html



① keras の ImageDataGenerator を使用する

② 前処理レイヤーを使用する（実験段階）

- https://www.tensorflow.org/tutorials/images/data_augmentation

③ カスタムレイヤーを追加する

- layers.Lambda を使用する

④ tf.image を使用する。

```python
def augment(image,label):
  image, label = resize_and_rescale(image, label)
  # Add 6 pixels of padding
  image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6) 
   # Random crop back to the original size
  image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
  image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
  image = tf.clip_by_value(image, 0, 1)
  return image, label

train_ds = (
    train_ds
    .shuffle(1000)
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
) 
```





### 転移学習・ファインチューニング（TF Hub）





### Callbacks





### @tf.function

つけるつけないで速度を測って、本当に早くなるか検証。(CIFAR10)



### 全部盛りの実装サンプル（MNIST）



SubclassingAPI

- 重み初期化
- Early Stopping
- 転移学習
- Data Augmentation
- 分散学習
- カスタム活性化関数
- バッチノーマリゼーション
- ドロップアウト

```python
import numpy as np 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf 
from tensorflow.keras import datasets, layers, models, optimizers, losses, metrics 

# データの準備
mnist = datasets.mnist 
(x_train_val, t_train_val), (x_test, t_test) = mnist.load_data() 
x_train, x_val, t_train, t_val = train_test_split(x_train_val, t_train_val, test_size=0.3, random_state=0)

# カスタム活性化関数
def swish(x, beta=1):
    return x * tf.nn.sigmoid(beta * x)

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=0, verbose=0):
        self._step = 0 
        self._loss = float('inf')
        self.patience = patience 
        self.verbose = verbose 
    def __call__(self, loss):
        if self._loss < loss:
            self._step += 1 
            if self._step > self.patience:
                if self.verbose:
                    print("early stopping")
                return True 
        else:
            self._step = 0
            self._loss = loss
        return False

# モデル
class Net(models.Model):
    def __init__(self, hidden_dim, output_dim):
        super().__init__() 
        self.model = models.Sequential([
            layers.BatchNormalization(),
            layers.Conv2D(hidden_dim, kernel_size=3, activation=swish, kernel_initializer='he_normal'), 
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(output_dim, activation="softmax")
        ])
    def call(self, x):
        return self.model(x)

net = Net(64, 10)
es = EarlyStopping()

criterion = losses.SparseCategoricalCrossentropy()
optimizer = optimizers.SGD(learning_rate=1e-3, momentum=0.9, nesterov=True)
train_loss = metrics.Mean()
train_acc = metrics.SparseCategoricalAccuracy() 
test_loss = metrics.Mean()
test_acc = metrics.SparseCategoricalAccuracy() 


def train(x, t):
    # 勾配算出するスコープを指定
    with tf.GradientTape() as tape:
        preds = net(x)
        loss = criterion(t, preds)
    # 逆伝播
    grads = tape.gradient(loss, net.trainable_variables)
    # 更新
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    train_loss(loss)
    train_acc(t, preds)
    return loss 

def test(x, t):
    preds = net(x)
    loss = criterion(t, preds)
    test_loss(loss)
    test_acc(t, preds)

epochs = 30
batch_size=256
n_batches_train = x_train.shape[0] // batch_size 
n_batches_val = x_val.shape[0] // batch_size 
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

# 学習
for epoch in range(epochs):
    x, t = shuffle(x_train, t_train)
    # 学習
    for batch in range(n_batches_train):
        start = batch * batch_size 
        emd = start + batch_size 
        train(x[start:end], t[start:end])
    # 検証
    for batch in range(n_batches_val):
        start = batch * batch_size 
        emd = start + batch_size 
        test(x_val[start:end], t_val[start:end])
    #　ログ
    history['train_loss'].append(train_loss.result())
    history['train_acc'].append(train_acc.result())
    history['val_loss'].append(val_loss.result())
    history['val_acc'].append(val_acc.result())
    # Early Stopping
    if ea(val_loss): 
        break

# テスト
test(x_test, t_test)

```





##  memo 



- WideResNet
  - 深く thin なモデルよりも、浅く wide なモデルのほうが良いのでは？
  - 16 層の WideResNet が 1000層の ResNetと比較して、同等の精度およびパラメータ数で数倍早く学習できる。

- ResNeXt
  - residual (bottleneck) モジュールの中身を分岐させて並列に処理し、最後に和をとる
  - cardinality ; 何個に並列させるか
  - grouped convolution
- Xecption
  - depthwise separable conv に代替
- EfficientNet
  - どの大きさのモデルのときにどの解像度にするの？
- SK-Net
  - Grouped−Conv → SEブロック
- ResNeSt
- NASNet





https://qiita.com/yu4u/items/7e93c454c9410c4b5427



