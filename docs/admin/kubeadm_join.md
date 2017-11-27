
Run this on any machine you wish to join an existing cluster

### Synopsis



When joining a kubeadm initialized cluster, we need to establish
bidirectional trust. This is split into discovery (having the Node
trust the Kubernetes Master) and TLS bootstrap (having the Kubernetes
Master trust the Node).

There are 2 main schemes for discovery. The first is to use a shared
token along with the IP address of the API server. The second is to
provide a file - a subset of the standard kubeconfig file. This file
can be a local file or downloaded via an HTTPS URL. The forms are
kubeadm join --discovery-token abcdef.1234567890abcdef 1.2.3.4:6443,
kubeadm join --discovery-file path/to/file.conf, or kubeadm join
--discovery-file https://url/file.conf. Only one form can be used. If
the discovery information is loaded from a URL, HTTPS must be used.
Also, in that case the host installed CA bundle is used to verify
the connection.

If you use a shared token for discovery, you should also pass the
--discovery-token-ca-cert-hash flag to validate the public key of the
root certificate authority (CA) presented by the Kubernetes Master. The
value of this flag is specified as "<hash-type>:<hex-encoded-value>",
where the supported hash type is "sha256". The hash is calculated over
the bytes of the Subject Public Key Info (SPKI) object (as in RFC7469).
This value is available in the output of "kubeadm init" or can be
calcuated using standard tools. The --discovery-token-ca-cert-hash flag
may be repeated multiple times to allow more than one public key.

If you cannot know the CA public key hash ahead of time, you can pass
the --discovery-token-unsafe-skip-ca-verification flag to disable this
verification. This weakens the kubeadm security model since other nodes
can potentially impersonate the Kubernetes Master.

The TLS bootstrap mechanism is also driven via a shared token. This is
used to temporarily authenticate with the Kubernetes Master to submit a
certificate signing request (CSR) for a locally created key pair. By
default, kubeadm will set up the Kubernetes Master to automatically
approve these signing requests. This token is passed in with the
--tls-bootstrap-token abcdef.1234567890abcdef flag.

Often times the same token is used for both parts. In this case, the
--token flag can be used instead of specifying each token individually.


```
kubeadm join [flags]
```

### Options

```
      --config string                                 Path to kubeadm config file.
      --cri-socket string                             Specify the CRI socket to connect to. (default "/var/run/dockershim.sock")
      --discovery-file string                         A file or url from which to load cluster information.
      --discovery-token string                        A token used to validate cluster information fetched from the master.
      --discovery-token-ca-cert-hash stringSlice      For token-based discovery, validate that the root CA public key matches this hash (format: "<type>:<value>").
      --discovery-token-unsafe-skip-ca-verification   For token-based discovery, allow joining without --discovery-token-ca-cert-hash pinning.
      --feature-gates string                          A set of key=value pairs that describe feature gates for various features. Options are:
CoreDNS=true|false (ALPHA - default=false)
DynamicKubeletConfig=true|false (ALPHA - default=false)
HighAvailability=true|false (ALPHA - default=false)
SelfHosting=true|false (BETA - default=false)
StoreCertsInSecrets=true|false (ALPHA - default=false)
      --ignore-preflight-errors stringSlice           A list of checks whose errors will be shown as warnings. Example: 'IsPrivilegedUser,Swap'. Value 'all' ignores errors from all checks.
      --node-name string                              Specify the node name.
      --tls-bootstrap-token string                    A token used for TLS bootstrapping.
      --token string                                  Use this token for both discovery-token and tls-bootstrap-token.
```

