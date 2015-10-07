## Client/Server mode (EXPERIMENTAL)

### Getting Started

By default flannel runs without a central controller, utilizing etcd for coordination.
However, it can also be configured to run in client/server mode, where a special instance of the flannel daemon (the server) is the only one that communicates with etcd.
This setup offers the advantange of having only a single server directly connecting to etcd, with the rest of the flannel daemons (clients) accessing etcd via the server.
The server is completely stateless and does not assume that it has exclusive access to the etcd keyspace.
In the future this will be exploited to provide failover; currently, however, the clients accept only a single endpoint to which to connect.
The stateless server also makes it possible to run some nodes in client mode side-by-side with those connecting to etcd directly.

To run the flannel daemon in server mode, simply provide the `--listen` flag:
```
$ flanneld --listen=0.0.0.0:8888
```

To run the flannel daemon in client mode, use the `--remote` flag to point it to a flannel server instance:
```
$ flanneld --remote=10.0.0.3:8888
```

It is important to note that the server itself does not join the flannel network (i.e. it won't assign itself a subnet) -- it just satisfies requests from the clients.
As such, if the host running the flannel server also needs to participate in the overlay, it should start two instances of flannel - one in client mode and one in server mode.


### Systemd Socket Activation

The server mode supports [systemd socket activation](http://www.freedesktop.org/software/systemd/man/systemd.socket.html).
To request the use of socket activation, use the `--listen` flag:
```
$ flanneld --listen=fd://
```

This assumes that the listening socket is passed in via the default descriptor 3.
To specify a different descriptor number, such as 5, use the following form:
```
$ flanneld --listen=fd://5
```

### Use of SSL/TLS to secure client/server communication

By default, the communication between the client and server is unencrypted (uses HTTP).
Just like the link between flannel and etcd can be secured via SSL/TLS, so too can the link between the client and the server be encrypted using SSL/TLS.
You will need a CA certificate and also a private key, certificate pair for the server.
The server certificate must be signed by the corresponding CA.
The easiest way to get started is by using the [etcd-ca](https://github.com/coreos/etcd-ca) project:

```
# Create a new Certificate Authority (CA)
$ etcd-ca init

# Export the CA certifcate -- this will generate ca.crt
$ etcd-ca export | tar xv

# Create a new private key for the server
$ etcd-ca new-cert myserver

# Sign the server private key by the CA
$ etcd-ca sign myserver

# Export the server key and certifiate
# This will generate myserver.key and myserver.crt
$ etcd-ca export myserver | tar xv
```

You can now start the flannel server, specifying the private key and corresponding signed certificate:
```
$ flanneld --listen=0.0.0.0 --remote-certfile=./myserver.crt --remote-keyfile=./myserver.key
```

Finally, start the flannel client(s) pointing them at the CA certificate that was used to sign the server certificate:

```
$ flanneld --remote=10.0.0.3:8888 --remote-cafile=./ca.crt
```

### Authenticating clients by use of client certificates

You can use client SSL certificates to restrict connecting clients to those that have their certificate signed by your CA.
Using [etcd-ca](https://github.com/coreos/etcd-ca) as the CA, first make sure you have executed ran the steps in the previous section.
Next, generate and sign a certificate for the client (repeat steps below for each client):

```
# Create a private key for the client1
$ etcd-ca new-cert client1

# Sign the client1 private key by the CA
$ etcd-ca sign client1

# Export the client1 key and certifiate
# This will generate client1.key and client1.crt
$ etcd-ca export client1 | tar xv
```

Start the server, specifying the CA certificate that was used to sign the client certificates:
```
$ flanneld --listen=0.0.0.0 --remote-certfile=./myserver.crt --remote-keyfile=./myserver.key --remote-cafile=./ca.crt
```

Launch the clients by also specifying their private key and corresponding certificate:
```
$ flanneld --remote=10.0.0.3:8888 --remote-cafile=./ca.crt --remote-keyfile=./client1.key --remote-certfile=./client1.crt
```
