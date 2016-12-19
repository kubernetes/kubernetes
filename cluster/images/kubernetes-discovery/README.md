### kube-discovery

An initial implementation of a Kubernetes discovery service using JSON Web Signatures.

This prototype runs within Kubernetes itself.

## Requirements

This pod expects to be launched like this:
```
        - "--proxy-client-cert-file=/var/run/auth-proxy-client/tls.crt"
        - "--proxy-client-key-file=/var/run/auth-proxy-client/tls.key"
        - "--tls-cert-file=/var/run/serving-cert/tls.crt"
        - "--tls-private-key-file=/var/run/serving-cert/tls.key"
        - "--tls-ca-file=/var/run/serving-ca/ca.crt"
        - "--client-ca-file=/var/run/client-ca/ca.crt"
        - "--requestheader-username-headers=X-Remote-User"
        - "--requestheader-group-headers=X-Remote-Group"
        - "--requestheader-extra-headers-prefix=X-Remote-Extra-"
        - "--requestheader-client-ca-file=/var/run/request-header-ca/ca.crt"
        - "--etcd-servers=https://etcd.kube-public.svc:4001"
        - "--etcd-certfile=/var/run/etcd-client-cert/tls.crt"
        - "--etcd-keyfile=/var/run/etcd-client-cert/tls.key"
        - "--etcd-cafile=/var/run/etcd-ca/ca.crt"
```

And example is located in https://github.com/kubernetes/kubernetes/blob/master/hack/local-up-discovery.sh.
