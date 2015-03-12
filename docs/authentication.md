# Authentication Plugins

Kubernetes can accept passwords, bearer tokens, or negotiate authentication to
authenticate users for API calls.

Password authentication is enabled by passing the `--saslauthd=PARAMS` option to
apiserver.  The parameters are a comma-separated list of three items: the
default realm name to pass to saslauthd, the location of the saslauthd socket,
and the SASL service name to pass to saslauthd.  If the socket location is not
specified or is left empty, `/var/run/saslauthd/mux` will be assumed.  If the
SASL service name is not specified or is left empty, a default will constructed
from the name of the binary (currently `kube-apiserver`).  The saslauthd service
must be configured separately, but often defaults to attempting to check a
password using PAM, using the SASL service name as the PAM service name.

Token authentication is enabled by passing the `--token_auth_file=SOMEFILE`
option to apiserver.  Currently, tokens last indefinitely, and the token list
cannot be changed without restarting apiserver.  We plan in the future for
tokens to be short-lived, and to be generated as needed rather than stored in a
file.

The token file format is implemented in `plugin/pkg/auth/authenticator/token/tokenfile/...`
and is a csv file with 3 columns: token, user name, user uid.

Negotiate authentication is enabled by passing the `--gssproxy=SOCKET` option to
apiserver.  The gss-proxy daemon should be configured to allow clients which are
running as the UID which the kube-apiserver runs as, and which connect to that
socket, to use krb5 credentials stored in a keytab to act as a GSSAPI acceptor
(service), for example by adding to gssproxy.conf:

	[service/kube-apiserver]
	 mechs = krb5
	 socket = /run/gssproxy-kube-apiserver.sock
	 cred_store = keytab:/etc/kube.keytab
	 cred_usage = accept
	 euid = 0

The apiserver would then be invoked with `--gssproxy=/run/gssproxy-kube-apiserver.sock`,
and would be expected to have Kerberos keys for the
`HTTP/kubernetes-master@REALM` service present in its `/etc/kube.keytab` file.

## Plugin Development

We plan for the Kubernetes API server to issue tokens
after the user has been (re)authenticated by a *bedrock* authentication
provider external to Kubernetes.  We plan to make it easy to develop modules
that interface between kubernetes and a bedrock authentication provider (e.g.
github.com, google.com, enterprise directory, kerberos, etc.)
