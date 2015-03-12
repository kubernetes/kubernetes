Package gss provides bindings for a C implementation of GSS-API (specifically, MIT Kerberos 1.12 or later) using cgo.  The provided API is relatively stable but still subject to change.

To download and build:
```
go get github.com/nalind/gss/...
```

In broad strokes:
* gss\_buffer\_t is replaced by either []byte or string
* OIDs and OID sets are passed around as encoding/asn1 ObjectIdentifiers and arrays of encoding/asn1 ObjectIdentifiers
* memory management is still very much done manually

Package gss/proxy provides a client for [gss-proxy](https://fedorahosted.org/gss-proxy/).  The provided API is relatively stable but still subject to change, particularly around name attributes.
* OIDs and OID sets are passed around as encoding/asn1 ObjectIdentifiers and arrays of encoding/asn1 ObjectIdentifiers
* The single Release RPC is replaced with two wrappers: ReleaseCred and ReleaseSecCtx.
* The proxy doesn't currently allow use of SPNEGO "credentials", so a minimal SPNEGO implementation is added here.

In order to use the proxy, your /etc/gssproxy/gssproxy.conf will need a stanza which the proxy will use to decide which credentials your process will be able to access, and over which socket it will be able to use them:

```
[service/proxy-clients]
  mechs = krb5
  euid = 0
  allow_any_uid = yes
  socket = /run/gssproxy-clients.sock
  cred_store = ccache:KEYRING:persistent:%U
```

Likewise, your server will need a stanza of its own:

```
[service/http-server]
  mechs = krb5
  euid = 48
  socket = /run/gssproxy-http.sock
  cred_store = keytab:/etc/httpd/conf/httpd.keytab
```
