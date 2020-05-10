# 日本語化

## spacyを日本語対応させる
### GiNZAのインストール

```Console
pip install -U ginza
```

## データセットの作成
### データセットの用意

```Console
mydata
├── domain_a.txt
└── domain_b.txt
```

- domain_a.txt
    - 変換元のテキスト。1行1文で記述する。
- domain_b.txt
    - 変換先のテキスト。1行1文で記述する。

A→B、B→Aのどちら方向の変換もできるので、A/Bのどっちが変換元/変換先でもいい。

### データのビルド

```Console
python buildJaData.py
```

## 学習

```Console
python trainNetwork.py --lang_type ja
```