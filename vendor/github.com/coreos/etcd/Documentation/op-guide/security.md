# Security model

etcd supports automatic TLS as well as authentication through client certificates for both clients to server as well as peer (server to server / cluster) communication.

To get up and running, first have a CA certificate and a signed key pair for one member. It is recommended to create and sign a new key pair for every member in a cluster.

For convenience, the [cfssl] tool provides an easy interface to certificate generation, and we provide an example using the tool [here][tls-setup]. Alternatively, try this [guide to generating self-signed key pairs][tls-guide].

## Basic setup

etcd takes several certificate related configuration options, either through command-line flags or environment variables:

**Client-to-server communication:**

`--cert-file=<path>`: Certificate used for SSL/TLS connections **to** etcd. When this option is set, advertise-client-urls can use the HTTPS schema.

`--key-file=<path>`: Key for the certificate. Must be unencrypted.

`--client-cert-auth`: When this is set etcd will check all incoming HTTPS requests for a client certificate signed by the trusted CA, requests that don't supply a valid client certificate will fail.

`--trusted-ca-file=<path>`: Trusted certificate authority.

`--auto-tls`: Use automatically generated self-signed certificates for TLS connections with clients.

**Peer (server-to-server / cluster) communication:**

The peer options work the same way as the client-to-server options:

`--peer-cert-file=<path>`: Certificate used for SSL/TLS connections between peers. This will be used both for listening on the peer address as well as sending requests to other peers.

`--peer-key-file=<path>`: Key for the certificate. Must be unencrypted.

`--peer-client-cert-auth`: When set, etcd will check all incoming peer requests from the cluster for valid client certificates signed by the supplied CA.

`--peer-trusted-ca-file=<path>`: Trusted certificate authority.

`--peer-auto-tls`: Use automatically generated self-signed certificates for TLS connections between peers.

If either a client-to-server or peer certificate is supplied the key must also be set. All of these configuration options are also available through the environment variables, `ETCD_CA_FILE`, `ETCD_PEER_CA_FILE` and so on.

## Example 1: Client-to-server transport security with HTTPS

For this, have a CA certificate (`ca.crt`) and signed key pair (`server.crt`, `server.key`) ready.

Let us configure etcd to provide simple HTTPS transport security step by step:

```sh
$ etcd --name infra0 --data-dir infra0 \
  --cert-file=/path/to/server.crt --key-file=/path/to/server.key \
  --advertise-client-urls=https://127.0.0.1:2379 --listen-client-urls=https://127.0.0.1:2379
```

This should start up fine and it will be possible to test the configuration by speaking HTTPS to etcd:

```sh
$ curl --cacert /path/to/ca.crt https://127.0.0.1:2379/v2/keys/foo -XPUT -d value=bar -v
```

The command should show that the handshake succeed. Since we use self-signed certificates with our own certificate authority, the CA must be passed to curl using the `--cacert` option. Another possibility would be to add the CA certificate to the system's trusted certificates directory (usually in `/etc/pki/tls/certs` or `/etc/ssl/certs`).

**OSX 10.9+ Users**: curl 7.30.0 on OSX 10.9+ doesn't understand certificates passed in on the command line.
Instead, import the dummy ca.crt directly into the keychain or add the `-k` flag to curl to ignore errors.
To test without the `-k` flag, run `open ./fixtures/ca/ca.crt` and follow the prompts.
Please remove this certificate after testing!
If there is a workaround, let us know.

## Example 2: Client-to-server authentication with HTTPS client certificates

For now we've given the etcd client the ability to verify the server identity and provide transport security. We can however also use client certificates to prevent unauthorized access to etcd.

The clients will provide their certificates to the server and the server will check whether the cert is signed by the supplied CA and decide whether to serve the request.

The same files mentioned in the first example are needed for this, as well as a key pair for the client (`client.crt`, `client.key`) signed by the same certificate authority.

```sh
$ etcd --name infra0 --data-dir infra0 \
  --client-cert-auth --trusted-ca-file=/path/to/ca.crt --cert-file=/path/to/server.crt --key-file=/path/to/server.key \
  --advertise-client-urls https://127.0.0.1:2379 --listen-client-urls https://127.0.0.1:2379
```

Now try the same request as above to this server:

```sh
$ curl --cacert /path/to/ca.crt https://127.0.0.1:2379/v2/keys/foo -XPUT -d value=bar -v
```

The request should be rejected by the server:

```
...
routines:SSL3_READ_BYTES:sslv3 alert bad certificate
...
```

To make it succeed, we need to give the CA signed client certificate to the server:

```sh
$ curl --cacert /path/to/ca.crt --cert /path/to/client.crt --key /path/to/client.key \
  -L https://127.0.0.1:2379/v2/keys/foo -XPUT -d value=bar -v
```

The output should include:

```
...
SSLv3, TLS handshake, CERT verify (15):
...
TLS handshake, Finished (20)
```

And also the response from the server:

```json
{
    "action": "set",
    "node": {
        "createdIndex": 12,
        "key": "/foo",
        "modifiedIndex": 12,
        "value": "bar"
    }
}
```

## Example 3: Transport security & client certificates in a cluster

etcd supports the same model as above for **peer communication**, that means the communication between etcd members in a cluster.

Assuming we have our `ca.crt` and two members with their own keypairs (`member1.crt` & `member1.key`, `member2.crt` & `member2.key`) signed by this CA, we launch etcd as follows:


```sh
DISCOVERY_URL=... # from https://discovery.etcd.io/new

# member1
$ etcd --name infra1 --data-dir infra1 \
  --peer-client-cert-auth --peer-trusted-ca-file=/path/to/ca.crt --peer-cert-file=/path/to/member1.crt --peer-key-file=/path/to/member1.key \
  --initial-advertise-peer-urls=https://10.0.1.10:2380 --listen-peer-urls=https://10.0.1.10:2380 \
  --discovery ${DISCOVERY_URL}

# member2
$ etcd --name infra2 --data-dir infra2 \
  --peer-client-cert-auth --peer-trusted-ca-file=/path/to/ca.crt --peer-cert-file=/path/to/member2.crt --peer-key-file=/path/to/member2.key \
  --initial-advertise-peer-urls=https://10.0.1.11:2380 --listen-peer-urls=https://10.0.1.11:2380 \
  --discovery ${DISCOVERY_URL}
```

The etcd members will form a cluster and all communication between members in the cluster will be encrypted and authenticated using the client certificates. The output of etcd will show that the addresses it connects to use HTTPS.

## Example 4: Automatic self-signed transport security

For cases where communication encryption, but not authentication, is needed, etcd supports encrypting its messages with automatically generated self-signed certificates. This simplifies deployment because there is no need for managing certificates and keys outside of etcd.

Configure etcd to use self-signed certificates for client and peer connections with the flags `--auto-tls` and `--peer-auto-tls`:

```sh
DISCOVERY_URL=... # from https://discovery.etcd.io/new

# member1
$ etcd --name infra1 --data-dir infra1 \
  --auto-tls --peer-auto-tls \
  --initial-advertise-peer-urls=https://10.0.1.10:2380 --listen-peer-urls=https://10.0.1.10:2380 \
  --discovery ${DISCOVERY_URL}

# member2
$ etcd --name infra2 --data-dir infra2 \
  --auto-tls --peer-auto-tls \
  --initial-advertise-peer-urls=https://10.0.1.11:2380 --listen-peer-urls=https://10.0.1.11:2380 \
  --discovery ${DISCOVERY_URL}
```

Self-signed certificates do not authenticate identity so curl will return an error:

```sh
curl: (60) SSL certificate problem: Invalid certificate chain
```

To disable certificate chain checking, invoke curl with the `-k` flag:

```sh
$ curl -k https://127.0.0.1:2379/v2/keys/foo -Xput -d value=bar -v
```

## Notes for etcd proxy

etcd proxy terminates the TLS from its client if the connection is secure, and uses proxy's own key/cert specified in `--peer-key-file` and `--peer-cert-file` to communicate with etcd members.

The proxy communicates with etcd members through both the `--advertise-client-urls` and `--advertise-peer-urls` of a given member. It forwards client requests to etcd members’ advertised client urls, and it syncs the initial cluster configuration through etcd members’ advertised peer urls.

When client authentication is enabled for an etcd member, the administrator must ensure that the peer certificate specified in the proxy's `--peer-cert-file` option is valid for that authentication. The proxy's peer certificate must also be valid for peer authentication if peer authentication is enabled.

## Frequently asked questions

### I'm seeing a SSLv3 alert handshake failure when using TLS client authentication?

The `crypto/tls` package of `golang` checks the key usage of the certificate public key before using it.
To use the certificate public key to do client auth, we need to add `clientAuth` to `Extended Key Usage` when creating the certificate public key.

Here is how to do it:

Add the following section to openssl.cnf:

```
[ ssl_client ]
...
  extendedKeyUsage = clientAuth
...
```

When creating the cert be sure to reference it in the `-extensions` flag:

```
$ openssl ca -config openssl.cnf -policy policy_anything -extensions ssl_client -out certs/machine.crt -infiles machine.csr
```

### With peer certificate authentication I receive "certificate is valid for 127.0.0.1, not $MY_IP"
Make sure to sign the certificates with a Subject Name the member's public IP address. The `etcd-ca` tool for example provides an `--ip=` option for its `new-cert` command.

The certificate needs to be signed for the member's FQDN in its Subject Name, use Subject Alternative Names (short IP SANs) to add the IP address. The `etcd-ca` tool provides `--domain=` option for its `new-cert` command, and openssl can make [it][alt-name] too.

[cfssl]: https://github.com/cloudflare/cfssl
[tls-setup]: ../../hack/tls-setup
[tls-guide]: https://github.com/coreos/docs/blob/master/os/generate-self-signed-certificates.md
[alt-name]: http://wiki.cacert.org/FAQ/subjectAltName
