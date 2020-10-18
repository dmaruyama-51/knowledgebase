# さくらVPS でセキュリティ設定



login : ubuntu

> password : ------------------



- ユーザー追加 & sudo 権限付与

  ```
  sudo adduser <user_name>
  
  sudo gpasswd -a <user_name> sudo
  ```

- ユーザーの切り替え

  ```
  sudo su - <user_name>
  ```

- vim, curl のインストール

  ```
  sudo apt-get update
  
  sudo apt-get install vim 
  
  sudo apt-get install curl
  ```

- ssh 接続

  ```
  sudo apt-get install ssh
  ```

  - 公開鍵認証設定（mac）

  ```
  ssh-kyegen
  
  # config ファイルを vim で編集
  vim ~/.ssh/config
  ```

  - ```
    Host <任意のhostname> 
    	HostName <IPアドレス>
    	User <user_name>
    	Port 22 
    	IdentityFiule ~/.ssh/**** # 作成した秘密鍵のPATH
    ```

  - 公開鍵認証設定（Ubuntu）

  ```
  mkdir ~/.ssh
  chmod 700 ~/.ssh # 権限
  
  vim ~/.ssh/autorized_keys # 公開鍵の情報を貼り付けて保存 ssh-rsa ~~
  chmod 600 ~/.ssh/authorized_keys
  ```

  → ssh <hostname> でログインできる！

- セキュリティ対策

  - root ユーザーのログインの禁止

  - パスワード認証の禁止 ※ root ユーザーでログインすること（ubuntu）

    ```
    sudo vim /etc/ssh/sshd_config 
    ```

    以下に書き換え

    ```
    PermitRootLogin no
    
    PasswordAuthentication no
    ```

  - ポート番号の変更（※ 例えば 56789）

    ```
    vim iptables.rules
    ```

    ```
    *filter
    :INPUT ACCEPT [0:0]
    :FORWARD ACCEPT [0:0]
    :OUTPUT ACCEPT [0:0]
    -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
    -A INPUT -p icmp -j ACCEPT
    -A INPUT -i lo -j ACCEPT
    # <- コメントアウト
    #-A INPUT -m state --state NEW -m tcp -p tcp --dport 22 -j ACCEPT
    # <- 追記
    -A INPUT -m state --state NEW -m tcp -p tcp --dport 56789 -j ACCEPT
    -A INPUT -j REJECT --reject-with icmp-host-prohibited
    -A FORWARD -j REJECT --reject-with icmp-host-prohibited
    COMMIT
    ```

    ```
    # 変更の反映
    sudo iptables-restore < iptables.rules
    
    # 現在適用されているルールの確認
    sudo iptables -L -n -v
    ```

    ssh 設定ファイルを書き換える（`sudo vim /etc/ssh/sshd_config `）

    ```
    #Port 22
    Port 56789
    ```

    変更の反映

    ```
    sudo /etc/init.d/ssh restart
    ```

    mac 側のポート番号を書き換え（`vim ~/.ssh/config`）

    ```
    Host <任意のhostname> 
    	HostName <IPアドレス>
    	User <user_name>
    	# Port 22 
    	Port 56789
    	IdentityFiule ~/.ssh/**** 
    ```

    

    