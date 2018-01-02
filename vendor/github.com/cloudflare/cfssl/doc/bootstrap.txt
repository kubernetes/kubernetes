Bootstrapping CFSSL
====================

CFSSL has no other dependencies besides a working Go 1.4 installation.
It uses only standard library components, besides those packages
included in the software.

1. Installing CFSSL

```
go get -u github.com/cloudflare/cfssl/cmd/cfssl
```

The `cfssl` binary may now be found in the `$GOPATH/bin` directory.

1.1 Installing mkbundle

Installing the `mkbundle` utility is similar:

```
go get -u github.com/cloudflare/cfssl/cmd/mkbundle
```

1.2 Installing cfssljson

The `cfssljson` utility is installed with:

```
go get -u github.com/cloudflare/cfssl/cmd/cfssljson
```

Alternatively, all three can be accomplished in one pass:

```
go get -u github.com/cloudflare/cfssl/cmd/...
```

All three binaries will now be in the `$GOPATH/bin` directory.

2. Set up the intermediate and root certificate bundles

The pre-built default CloudFlare bundles may be found in the
[cfssl_trust](https://github.com/cloudflare/cfssl_trust) repository.

`cfssl` will, by default, look for these bundles in `/etc/cfssl/`;
it will look for a `ca-bundle.crt` and `int-bundle.crt`.

3. [Optional] Set up the CA certificate and key

First, create a JSON file containing the key request similar to the
following (perhaps in `ca.json`):

```
{
	"hosts": [
		"ca.example.com"
	],
	"key": {
		"algo": "rsa",
		"size": 4096
	},
	"names": [
		{
			"C": "US",
			"L": "San Francisco",
			"O": "Internet Widgets, LLC",
			"OU": "Certificate Authority",
			"ST": "California"
		}
	]
}
```

Then, initialise the CA:

```
cfssl genkey -initca ca.json | cfssljson -bare ca
```

When `cfssl` starts up, it will look by default for a CA key named
`ca-key.pem` and a certificate named `ca.pem` in `/etc/cfssl`; this may
be changed via the command line options. If it can't find the key and
certificate mentioned, it start up without the CA functionality enabled.

4. Start up the server

```
cfssl serve
```

The endpoints for the server are described in `doc/api.txt`.
