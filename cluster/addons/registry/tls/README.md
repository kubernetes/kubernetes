# Enable TLS for Kube-Registry 

This document describes how to enable TLS for kube-registry. Before you start, please check if you have all the prerequisite:

- A domain for kube-registry. Assuming it is ` myregistrydomain.com`.
- Domain certificate and key. Assuming they are `domain.crt` and `domain.key`

### Pack domain.crt and domain.key into a Secret 

```console
$ kubectl --namespace=kube-system create secret generic registry-tls-secret --from-file=domain.crt=domain.crt --from-file=domain.key=domain.key
```

### Run Registry

Please be noted that this sample rc is using emptyDir as storage backend for simplicity. 

<!-- BEGIN MUNGE: EXAMPLE registry-tls-rc.yaml -->
```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: kube-registry-v0
  namespace: kube-system
  labels:
    k8s-app: kube-registry
    version: v0
#    kubernetes.io/cluster-service: "true"
spec:
  replicas: 1
  selector:
    k8s-app: kube-registry
    version: v0
  template:
    metadata:
      labels:
        k8s-app: kube-registry
        version: v0
#        kubernetes.io/cluster-service: "true"
    spec:
      containers:
      - name: registry
        image: registry:2
        resources:
          # keep request = limit to keep this container in guaranteed class
          limits:
            cpu: 100m
            memory: 100Mi
          requests:
            cpu: 100m
            memory: 100Mi
        env:
        - name: REGISTRY_HTTP_ADDR
          value: :5000
        - name: REGISTRY_STORAGE_FILESYSTEM_ROOTDIRECTORY
          value: /var/lib/registry
        - name: REGISTRY_HTTP_TLS_CERTIFICATE
          value: /certs/domain.crt
        - name: REGISTRY_HTTP_TLS_KEY
          value: /certs/domain.key
        volumeMounts:
        - name: image-store
          mountPath: /var/lib/registry
        - name: cert-dir
          mountPath: /certs
        ports:
        - containerPort: 5000
          name: registry
          protocol: TCP
      volumes:
      - name: image-store
        emptyDir: {}
      - name: cert-dir
        secret:
          secretName: registry-tls-secret
```
<!-- END MUNGE: EXAMPLE registry-tls-rc.yaml -->

### Expose External IP for Kube-Registry

Modify the default kube-registry service to `LoadBalancer` type and point the DNS record of `myregistrydomain.com` to the service external ip. 

<!-- BEGIN MUNGE: EXAMPLE registry-tls-svc.yaml -->
```yaml
apiVersion: v1
kind: Service
metadata:
  name: kube-registry
  namespace: kube-system
  labels:
    k8s-app: kube-registry
#    kubernetes.io/cluster-service: "true"
    kubernetes.io/name: "KubeRegistry"
spec:
  selector:
    k8s-app: kube-registry
  type: LoadBalancer
  ports:
  - name: registry
    port: 5000
    protocol: TCP
```
<!-- END MUNGE: EXAMPLE registry-tls-svc.yaml -->

### To Verify 

Now you should be able to access your kube-registry from another docker host. 
```console
docker pull busybox
docker tag busybox myregistrydomain.com:5000/busybox
docker push myregistrydomain.com:5000/busybox
docker pull myregistrydomain.com:5000/busybox
```


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/registry/tls/README.md?pixel)]()
