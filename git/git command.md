# git 



## SSH Key 



```
cd ~/

ssh-keygen

cat ~/.ssh/id_rsa.pub # 公開鍵を確認

cat ~/.ssh/id_rsa # 秘密鍵を確認
```



- 公開鍵 -> 自身のアカウントの SSH に登録
- 秘密鍵 -> 各レポジトリの Secret に登録





## branch 

```
git branch -a # branch 一覧の確認

git branch <branch名> # branch をローカルに作成

git checkout <branch名> # branch に移動

git push -u origin <branch名> # branch をリモートに反映
```

