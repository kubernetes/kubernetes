# Transport Package Examples

`maserver` and `maclient` are a mutually authenticated server and client;
the client will connect to the server and send a few messages.

## Set up

A running CFSSL is needed. The `genca.sh` script should generate
everything that's needed to run CFSSL locally. In a terminal for the
CA:

```
$ basename $(pwd)
example
$ ./genca.sh
2015/10/27 14:00:29 [INFO] generating a new CA key and certificate from CSR
2015/10/27 14:00:29 [INFO] generate received request
2015/10/27 14:00:29 [INFO] received CSR
2015/10/27 14:00:29 [INFO] generating key: rsa-4096
2015/10/27 14:00:32 [INFO] encoded CSR
2015/10/27 14:00:33 [INFO] signed certificate with serial number 2940131150448804266
$ cfssl serve -ca ca.pem -ca-key ca-key.pem -config config.json
...
2015/10/27 14:00:35 [INFO] Setting up '/api/v1/cfssl/sign' endpoint
```

The providing `config.json` contains the CFSSL configuration; the
`client.json` and `server.json` configurations are based on this
config.

## Running the server

The server expects a `server.json` in the same directory containing
the configuration. One is provided in the server source, or it may be
overridden using the `-f` command line flag.

```
$ basename $(pwd)
example
$ cd maserver/
$ go run server.go -a 127.0.0.1:9876
$ go run server.go -a 127.0.0.1:9876
2015/10/27 14:05:47 [INFO] using client auth
2015/10/27 14:05:47 [DEBUG] transport isn't ready; attempting to refresh keypair
2015/10/27 14:05:47 [DEBUG] key and certificate aren't ready, loading
2015/10/27 14:05:47 [DEBUG] failed to load keypair: open server.key: no such file or directory
2015/10/27 14:05:47 [DEBUG] transport's certificate is out of date (lifespan 0)
2015/10/27 14:05:47 [INFO] encoded CSR
2015/10/27 14:05:47 [DEBUG] requesting certificate from CA
2015/10/27 14:05:47 [DEBUG] giving the certificate to the provider
2015/10/27 14:05:47 [DEBUG] storing the certificate
2015/10/27 14:05:47 [INFO] setting up auto-update
2015/10/27 14:05:47 [INFO] listening on 127.0.0.1:9876
```

At this point, the clients can start talking to the server.

## Running a client

At this point, clients just connect and send a few messages, ensuring
the server acknowledges the messages. The client also expects a
`client.json` configuration in the same directory; once is provided in
the source directory, or it may be overridden using the `-f` command
line flag.

```
$ basename $(pwd)
example
$ go run client.go
2015/10/27 14:08:34 [DEBUG] transport isn't ready; attempting to refresh keypair
2015/10/27 14:08:34 [DEBUG] key and certificate aren't ready, loading
2015/10/27 14:08:34 [DEBUG] failed to load keypair: open client.key: no such file or directory
2015/10/27 14:08:34 [DEBUG] transport's certificate is out of date (lifespan 0)
2015/10/27 14:08:34 [INFO] encoded CSR
2015/10/27 14:08:34 [DEBUG] requesting certificate from CA
2015/10/27 14:08:34 [DEBUG] giving the certificate to the provider
2015/10/27 14:08:34 [DEBUG] storing the certificate
OK
$
```

## Auth Examples

The CA, server, and client ship with a `_auth.json` configuration file
that will use an authenticated CFSSL. The commands change to:

```
$ cfssl serve -ca ca.pem -ca-key ca-key.pem -config config_auth.json
$ go run server.go -a 127.0.0.1:9876 -f server_auth.json
$ go run client.go -f client_auth.json
```

