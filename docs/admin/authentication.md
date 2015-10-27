<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Authentication Plugins

Kubernetes uses client certificates, tokens, or http basic auth to authenticate users for API calls.

**Client certificate authentication** is enabled by passing the `--client-ca-file=SOMEFILE`
option to apiserver. The referenced file must contain one or more certificates authorities
to use to validate client certificates presented to the apiserver. If a client certificate
is presented and verified, the common name of the subject is used as the user name for the
request.

**Token File** is enabled by passing the `--token-auth-file=SOMEFILE` option
to apiserver.  Currently, tokens last indefinitely, and the token list cannot
be changed without restarting apiserver.

The token file format is implemented in `plugin/pkg/auth/authenticator/token/tokenfile/...`
and is a csv file with 3 columns: token, user name, user uid.

When using token authentication from an http client the apiserver expects an `Authorization`
header with a value of `Bearer SOMETOKEN`.

**OpenID Connect ID Token** is enabled by passing the following options to the apiserver:
- `--oidc-issuer-url` (required) tells the apiserver where to connect to the OpenID provider. Only HTTPS scheme will be accepted.
- `--oidc-client-id` (required) is used by apiserver to verify the audience of the token.
A valid [ID token](http://openid.net/specs/openid-connect-core-1_0.html#IDToken) MUST have this
client-id in its `aud` claims.
- `--oidc-ca-file` (optional) is used by apiserver to establish and verify the secure connection
to the OpenID provider.
- `--oidc-username-claim` (optional, experimental) specifies which OpenID claim to use as the user name. By default, `sub`
will be used, which should be unique and immutable under the issuer's domain. Cluster administrator can
choose other claims such as `email` to use as the user name, but the uniqueness and immutability is not guaranteed.

Please note that this flag is still experimental until we settle more on how to handle the mapping of the OpenID user to the Kubernetes user. Thus further changes are possible.

Currently, the ID token will be obtained by some third-party app. This means the app and apiserver
MUST share the `--oidc-client-id`.

Like **Token File**, when using token authentication from an http client the apiserver expects
an `Authorization` header with a value of `Bearer SOMETOKEN`.

**Basic authentication** is enabled by passing the `--basic-auth-file=SOMEFILE`
option to apiserver. Currently, the basic auth credentials last indefinitely,
and the password cannot be changed without restarting apiserver. Note that basic
authentication is currently supported for convenience while we finish making the
more secure modes described above easier to use.

The basic auth file format is implemented in `plugin/pkg/auth/authenticator/password/passwordfile/...`
and is a csv file with 3 columns: password, user name, user id.

When using basic authentication from an http client, the apiserver expects an `Authorization` header
with a value of `Basic BASE64ENCODED(USER:PASSWORD)`.

**Keystone authentication** is enabled by passing the `--experimental-keystone-url=<AuthURL>`
option to the apiserver during startup. The plugin is implemented in
`plugin/pkg/auth/authenticator/request/keystone/keystone.go`.
For details on how to use keystone to manage projects and users, refer to the
[Keystone documentation](http://docs.openstack.org/developer/keystone/). Please note that
this plugin is still experimental which means it is subject to changes.
Please refer to the [discussion](https://github.com/kubernetes/kubernetes/pull/11798#issuecomment-129655212)
and the [blueprint](https://github.com/kubernetes/kubernetes/issues/11626) for more details

## Plugin Development

We plan for the Kubernetes API server to issue tokens
after the user has been (re)authenticated by a *bedrock* authentication
provider external to Kubernetes.  We plan to make it easy to develop modules
that interface between Kubernetes and a bedrock authentication provider (e.g.
github.com, google.com, enterprise directory, kerberos, etc.)

## APPENDIX

### Creating Certificates

When using client certificate authentication, you can generate certificates manually or
using an existing deployment script.

**Deployment script** is implemented at
`cluster/saltbase/salt/generate-cert/make-ca-cert.sh`.
Execute this script with two parameters. First is the IP address of apiserver, the second is
a list of subject alternate names in the form `IP:<ip-address> or DNS:<dns-name>`.
The script will generate three files:ca.crt, server.crt and server.key.
Finally, add these parameters
`--client-ca-file=/srv/kubernetes/ca.crt`
`--tls-cert-file=/srv/kubernetes/server.cert`
`--tls-private-key-file=/srv/kubernetes/server.key`
into apiserver start parameters.

**easyrsa** can be used to manually generate certificates for your cluster.

1.  Download, unpack, and initialize the patched version of easyrsa3.

          curl -L -O https://storage.googleapis.com/kubernetes-release/easy-rsa/easy-rsa.tar.gz
          tar xzf easy-rsa.tar.gz
          cd easy-rsa-master/easyrsa3
          ./easyrsa init-pki
1.  Generate a CA. (`--batch` set automatic mode. `--req-cn` default CN to use.)

          ./easyrsa --batch "--req-cn=${MASTER_IP}@`date +%s`" build-ca nopass
1.  Generate server certificate and key.
    (build-server-full [filename]: Generate a keypair and sign locally for a client or server)

          ./easyrsa --subject-alt-name="IP:${MASTER_IP}" build-server-full kubernetes-master nopass
1.  Copy `pki/ca.crt`  `pki/issued/kubernetes-master.crt`
    `pki/private/kubernetes-master.key` to your directory.
1.  Remember fill the parameters
    `--client-ca-file=/yourdirectory/ca.crt`
    `--tls-cert-file=/yourdirectory/server.cert`
    `--tls-private-key-file=/yourdirectory/server.key`
    and add these into apiserver start parameters.

**openssl** can also be use to manually generate certificates for your cluster.

1.  Generate a ca.key with 2048bit
    `openssl genrsa -out ca.key 2048`
1.  According to the ca.key generate a ca.crt.  (-days set the certificate effective time).
    `openssl req -x509 -new -nodes -key ca.key -subj "/CN=${MASTER_IP}" -days 10000 -out ca.crt`
1.  Generate a server.key with 2048bit
    `openssl genrsa -out server.key 2048`
1.  According to the server.key generate a server.csr.
    `openssl req -new -key server.key -subj "/CN=${MASTER_IP}" -out server.csr`
1.  According to the ca.key, ca.crt and server.csr generate the server.crt.
    `openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt
    -days 10000`
1.  View the certificate.
    `openssl x509  -noout -text -in ./server.crt`
    Finally, do not forget fill the same parameters and add parameters into apiserver start parameters.



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/authentication.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
