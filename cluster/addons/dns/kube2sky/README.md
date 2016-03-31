# kube2sky
==============

A bridge between Kubernetes and SkyDNS.  This will watch the kubernetes API for
changes in Services and then publish those changes to SkyDNS through etcd.

For now, this is expected to be run in a pod alongside the etcd and SkyDNS
containers.

## Namespaces

Kubernetes namespaces become another level of the DNS hierarchy.  See the
description of `--domain` below.

## Flags

`--domain`: Set the domain under which all DNS names will be hosted.  For
example, if this is set to `kubernetes.io`, then a service named "nifty" in the
"default" namespace would be exposed through DNS as
"nifty.default.svc.kubernetes.io".

`--v`: Set logging level

`--etcd-mutation-timeout`: For how long the application will keep retrying etcd
mutation (insertion or removal of a dns entry) before giving up and crashing.

`--etcd-servers`: List of etcd servers that are being used by skydns.

`--etcd-cafile`: x509 PEM-encoded Certificate Authority file to use for certificate verification.

`--etcd-certfile`: x509 PEM-encoded SSL certificate to use for client to etcd communication.

`--etcd-keyfile`: x509 PEM-encoded SSL key to use for client to etcd communication.

`--kube-master-url`: URL of kubernetes master. Required if `--kubecfg_file` is not set.

`--kubecfg-file`: Path to kubecfg file that contains the master URL and tokens to authenticate with the master.

`--log-dir`: If non empty, write log files in this directory

`--logtostderr`: Logs to stderr instead of files

## Secure communication with etcd

New set of flags enable kube2sky to communicate with secure etcd instances.

Please refer to official documentation about [etcd security].

Here's an example ReplicationController configuration for setting up DNS with secure etcd:

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: kube-dns-v9
  namespace: kube-system
  labels:
    k8s-app: kube-dns
    version: v9
    kubernetes.io/cluster-service: "true"
spec:
  replicas: 1
  selector:
    k8s-app: kube-dns
    version: v9
  template:
    metadata:
      labels:
        k8s-app: kube-dns
        version: v9
        kubernetes.io/cluster-service: "true"
    spec:
      hostNetwork: true
      containers:
      - name: kube2sky
        image: gcr.io/google_containers/kube2sky:1.15
        imagePullPolicy: Always
        resources:
          limits:
            cpu: 100m
            memory: 50Mi
        args:
        # command = "/kube2sky"
        - --domain=cluster.local
        - --etcd-servers=https://127.0.0.1:2379
        - --etcd-cafile=/etc/ssl/certs/ca.pem
        - --etcd-certfile=/etc/ssl/certs/etcd.pem
        - --etcd-keyfile=/etc/ssl/certs/etcd-key.pem
        volumeMounts:
        - mountPath: /etc/ssl/certs
          name: ssl-certs-etcd
          readOnly: true
      - name: skydns
        image: gcr.io/google_containers/skydns:2015-10-13-8c72f8c
        imagePullPolicy: Always
        resources:
          limits:
            cpu: 100m
            memory: 50Mi
        args:
        # command = "/skydns"
        - -machines=https://127.0.0.1:2379
        - -addr=0.0.0.0:53
        - -ns-rotate=false
        - -ca-cert=/etc/ssl/certs/ca.pem
        - -tls-pem=/etc/ssl/certs/etcd.pem
        - -tls-key=/etc/ssl/certs/etcd-key.pem
        - -domain=cluster.local
        ports:
        - containerPort: 53
          name: dns
          protocol: UDP
        - containerPort: 53
          name: dns-tcp
          protocol: TCP
        volumeMounts:
        - mountPath: /etc/ssl/certs/
          name: ssl-certs-etcd
          readOnly: true
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
            scheme: HTTP
          initialDelaySeconds: 30
          timeoutSeconds: 5
        readinessProbe:
          httpGet:
            path: /healthz
            port: 8080
            scheme: HTTP
          initialDelaySeconds: 1
          timeoutSeconds: 5
      - name: healthz
        image: gcr.io/google_containers/exechealthz:1.1
        resources:
          limits:
            cpu: 10m
            memory: 20Mi
        args:
        - -cmd=nslookup kubernetes.default.svc.cluster.local 127.0.0.1 >/dev/null
        - -port=8080
        ports:
        - containerPort: 8080
          protocol: TCP
      volumes:
      - name: etcd-storage
        emptyDir: {}
      - name: ssl-certs-etcd
        hostPath:
          path: /etc/ssl/certs/
      dnsPolicy: Default
```

[//]: # (Footnotes and references)

[etcd security]: <https://github.com/coreos/etcd/blob/master/Documentation/security.md>

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/dns/kube2sky/README.md?pixel)]()
