## Libtrust TLS Config Demo

This program generates key pairs and trust files for a TLS client and server.

To generate the keys, run:

```
$ go run genkeys.go
```

The generated files are:

```
$ ls -l client_data/ server_data/
client_data/:
total 24
-rw-------  1 jlhawn  staff  281 Aug  8 16:21 private_key.json
-rw-r--r--  1 jlhawn  staff  225 Aug  8 16:21 public_key.json
-rw-r--r--  1 jlhawn  staff  275 Aug  8 16:21 trusted_hosts.json

server_data/:
total 24
-rw-r--r--  1 jlhawn  staff  348 Aug  8 16:21 trusted_clients.json
-rw-------  1 jlhawn  staff  281 Aug  8 16:21 private_key.json
-rw-r--r--  1 jlhawn  staff  225 Aug  8 16:21 public_key.json
```

The private key and public key for the client and server are stored in `private_key.json` and `public_key.json`, respectively, and in their respective directories. They are represented as JSON Web Keys: JSON objects which represent either an ECDSA or RSA private key. The host keys trusted by the client are stored in `trusted_hosts.json` and contain a mapping of an internet address, `<HOSTNAME_OR_IP>:<PORT>`, to a JSON Web Key which is a JSON object representing either an ECDSA or RSA public key of the trusted server. The client keys trusted by the server are stored in `trusted_clients.json` and contain an array of JSON objects which contain a comment field which can be used describe the key and a JSON Web Key which is a JSON object representing either an ECDSA or RSA public key of the trusted client.

To start the server, run:

```
$ go run server.go
```

This starts an HTTPS server which listens on `localhost:8888`. The server configures itself with a certificate which is valid for both `localhost` and `127.0.0.1` and uses the key from `server_data/private_key.json`. It accepts connections from clients which present a certificate for a key that it is configured to trust from the `trusted_clients.json` file and returns a simple 'hello' message.

To make a request using the client, run:

```
$ go run client.go
```

This command creates an HTTPS client which makes a GET request to `https://localhost:8888`. The client configures itself with a certificate using the key from `client_data/private_key.json`. It only connects to a server which presents a certificate signed by the key specified for the `localhost:8888` address from `client_data/trusted_hosts.json` and made to be used for the `localhost` hostname. If the connection succeeds, it prints the response from the server.

The file `gencert.go` can be used to generate PEM encoded version of the client key and certificate. If you save them to `key.pem` and `cert.pem` respectively, you can use them with `curl` to test out the server (if it is still running).

```
curl --cert cert.pem --key key.pem -k https://localhost:8888
``` 
