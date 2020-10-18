[TOC]

# MySQL



```
# start
mysql.server start 

# connect
mysql -u root 

SHOW DATABASES;

USE <DATABASE NAME>

SHOW TABALES;

quit 

mysql.server stop
```



## 概要

### データベースとは

- 大量のデータを登録して、整理して保存しておき、必要なときに必要なデータを探し出せるようにしているデータの集まり。あるいはデータを管理するソフトウェア
- 具体的にどんなデータ？
  - 会員情報
  - 商品情報
- データベース管理システム（DBMS）
  - MySQL
  - PostgreSQL
  - Oracle Database
  - Microsoft Office Access



- エンティティ：テーブルに含まれるデータ項目をまとめたもの（ユニークなカラム名を集めたもの）
- ER図：Entity Relationship 
- 主キー（PK）：レコードを一意に特定できるデータ項目
- 外部キー（FK）：他テーブルとの関連付けに用いるデータ項目



### メタ：コマンドの分類



**基本**

- CRUD操作
  - Create -> INSERT
  - Read -> SELECT
  - Update -> UPDATE
  - Delte -> DELETE
- データ抽出系
  - LIMIT : 取得データ件数を制限（重要）
  - DISTINCT : 重複除く
  - ORDER BY : 並び替え
  - WHERE : 条件検索
  - LIKE : パターンマッチング
    - % : 任意の0文字以上の文字列
    - _ : 任意の1文字
    - ●●を含む； %●●%
    - ●●で終わる：%●●
    - ●●で始まる：●●%
    - 文字列に % や _ が含まれている場合→ESCAPE句を用いる \%みたいにする。
  - BETWEEN : 範囲条件
    - 以上、以下で、より大きい、より小さいには対応しない
  - IN条件、NOT IN 条件
    - IN : ある値が、列挙した値のどれかと等しいかを判定。一個でも一致してればTRUE
    - NOT IN : ある値が列挙した値のどれとも一致しないことを判定。一個でも一致していればFALSE
    - ALL : 値リストとそれぞれ比較して、すべて真ならTRUE
  - GROUP BY + 集計関数
    - COUNT, SUM, AVG, MAX, MIN, VARIANCE, STDDEV
  - HAVING : 集計結果に条件指定
  - ROUND : 指定した桁数に数値を丸める
    - ROUND(12.15, 1) -> 12.2
    - ROUND(12.15, -1) -> 10 : 一の位を丸める



**実行順序**

1. FROM
2. WHERE
3. GROUP BY 
4. HAVING
5. SELECT
6. ORDER BY
7. LIMIT



**条件文**





### Tips

- SELECT, FROM, USER などは、SQLの機能として特別な意味をもつため、列名として使用するときは `` バッククオートで囲むこと。
- コーディング規約
  - 大文字：予約後、小文字：予約語意外
  - インデントを下げる、インデント幅は揃える
- シングルクォーテーションの中にシングルクォーテーション→ 2つ繰り返すと認識される
- ORDER BY の条件を2つ機能させる
  - , で2つの条件をつなげる　`ORDER BY yr DESC, winner ASC;`
  - 前から順番に条件指定される。 yr で並び替えた上で、各中身をwinner 順にする。



```
式 subject IN ('Physics' , Chemistry') の値は 0 または 1 として扱われる。

1984年の賞の　受賞者 winner と分野 subject を分野と受賞者の名前順で表示する。ただし化学 Chemistry と物理学 Physics は最後の方に表示する。
```



ORDER BY subeject IN ('Physics' , Chemistry') でPhysics, Chemistry は 1　それ以外は 0 という数字にして並び替えが行われる→ この数字に基づいて並び替えれば、Chemistry後ろに来る！



- サブクエリ：一つの値を返すように書くことを意識、WHERE や SELECT で使える（スカラとして使う場合）,FROM で使用することもできる。WITH句のほうが読みやすい。





## MySQL入門



### MySQLの特徴

- オープンソースで無料利用可
- 導入実績豊富で信頼感がある
- 大規模なデータ処理にも耐えうる
- トランザクション機能
- バックアップ、リストア機能
- ・・・



### 基本操作

**データベース一覧の確認（SHOW DATABASE）**

```mysql
# データベース一覧の確認
SHOW DATABASES;

# LIKEに依る条件指定
SHOW DATABASES LIKE パターン

# 先頭に "my" がつくデータベースを表示
SHOW DATABASES LIKE "my%";

# 末尾に "scheme" がつくデータベースを表示
SHOW DATABASES LIKE "%schema";
```



**データベースの作成（CREATE DATABASE）**

```mysql
CREATE DATABASE cooking;

# 同じ名前のデータベースが存在したときのエラーを表示しない
CREATE DATABASE IF NOT EXISTS cooking;

# 文字コードを指定してデータベースを作成する
CREATE DATABASE cooking CHARACTER SET utf8;

# ハイフンがついたデータベースを作成する際にはバッククオートでくくる
CREATE DATABASE `test-db`;
```



**データベースの削除（DROP DATABASE）**

```mysql
DROP DATABASE cooking;

# 指定した名前のDBが存在しないときのエラーを表示しない
DROP DATABASE IF EXISTS cooking;
```



**テーブルの作成**

```mysql
# 使用するデータベースの指定
USE cooking;

# 選択中のデータベースを確認
SELECT DATABASE();

# テーブルの作成
CREATE TABLE menus (id INT, name VARCHAR(100));

# テーブルの確認
SHOW TABLES;
```

| TYPE                                    | DETAIL |                                       |
| --------------------------------------- | ------ | ------------------------------------- |
| INT                                     | 整数   |                                       |
| DECIMAL（全体の桁数, 小数点以下の桁数） | 少数   | DECIMAL(5, 2) -> 123.45のような少数値 |
| VARCHAR（文字数）                       | 文字列 |                                       |
| DATETIME                                | 日時   |                                       |
| DATE                                    | 日付   |                                       |





**テーブルにデータを追加**

```mysql
INSERT INTO menus(id, name) VALUES(1, "ハンバーグ")
```



**レコードを更新（UPDATE）**

```mysql
UPDATE menus SET name = "カレー" WHERE id = 1;
```



**レコードを削除（DELETE）**

```mysql
DELETE FROM menus WHERE id = 1;
```





### mysql コマンド（コマンドライン）



```shell
mysql -h ホスト名 -u ユーザー名 -p パスワード
```

※デフォルトの設定値

ホスト名：localhost , ユーザー名：0DBC(Windows) , パスワード：なし



**MySQLサーバーに接続**

```shell
mysql \
-u root \ #rootユーザーで接続
-u root cooking \ #cookingという名前のデータベースを使用することをここで指定することもできる
-D test \ #データベース名を指定
-e "SHOW DATABASES;" \ #コマンドラインから直接SQLクエリを実行
-h "hogehoge" \ #ホスト名を指定
-p mypassword \ #パスワードを指定
```



**MySQLサーバーとの接続を終了する**

```shell
# どちらでもよい
exit
quit
```





### VIEW の作成

- ビューとは、実際のデータを保存しないテーブル
- ビューは、SELECT文そのものを保存している

- メリット
  - データを保存しないため、記憶装置の容量を節約できる
  - 頻繁に使うSELECT文を毎回書かずに呼び出せる



```sql
# ビューの作成
CREATE VIEW ビュー名 (<ビューの列名1>, <ビューの列名2>, ...)
AS 
<SELECT文>
```



- ビュー定義におけるselect文にはORDER BY句は使えない



```sql
DROP VIEW <ビュー名>
```





### サブクエリ

FROM の中身をサブクエリにするパターン（↑）：テーブルとして活用

WHERE の条件をサブクエリ使用するパターン（↓）：スカラとして活用

```sql
SELECT shohin_bunrui, cnt_shohin
    -> FROM (SELECT shohin_bunrui, COUNT(*) AS cnt_shohin
    -> FROM Shohin
    -> GROUP BY shohin_bunrui) AS ShohinSum;
```

```sql
SELECT shohin_id, shohin_mei, hanbai_tanka

  -> FROM shohin

  -> WHERE hanbai_tanka > (SELECT AVG(hanbai_tanka) FROM shohin);
```

SELECT の中身をサブクエリにするパターン：スカラとして活用

```sql
SELECT shohin_id, shohin_mei, hanbai_tanka, (SELECT AVG(hanbai_tanka) FROM Shohin) AS avg_tanka
    -> FROM Shohin;
```



### 相関サブクエリ

```sql
SELECT shohin_bunrui, shohin_mei, hanbai_tanka
    -> FROM Shohin as S1
    -> WHERE hanbai_tanka > (SELECT AVG(hanbai_tanka) FROM Shohin AS S2
    -> WHERE S1.shohin_bunrui = S2.shohin_bunrui
    -> GROUP BY shohin_bunrui);
```



### テーブルをローカルの csv ファイルから作成

```mysql
CREATE TABLE <テーブル名>

ALTER TABLE <テーブル名> ADD カラム名 VARCHAR(40) not null;

LOAD DATA LOCAL INFILE '~/desktop/sample.csv' 
INTO TABLE <テーブル名> 
FIELDS TERMINATED BY ',' 
IGNORE 1 LINES;

```



