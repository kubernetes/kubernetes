Keys in this directory are generated for testing purposes only.

They can be generated using:
```sh
openssl req -x509 -newkey {key_format} -keyout {name}.key -out {name}.crt -days 36500 -nodes -subj "/CN=kube-ca"
```
