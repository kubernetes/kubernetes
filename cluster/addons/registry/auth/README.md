# Enable Authentication with Htpasswd for Kube-Registry 

Docker registry support a few authentication providers. Full list of supported provider can be found [here](https://docs.docker.com/registry/configuration/#auth). This document describes how to enable authentication with htpasswd for kube-registry. 

### Prepare Htpasswd Secret

Please generate your own htpasswd file. Assuming the file you generated is `htpasswd`. 
Creating secret to hold htpasswd...
```console
$ kubectl --namespace=kube-system create secret generic registry-auth-secret --from-file=htpasswd=htpasswd
```

### Run Registry

Please be noted that this sample rc is using emptyDir as storage backend for simplicity. 

<!-- BEGIN MUNGE: EXAMPLE registry-auth-rc.yaml -->
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
        - name: REGISTRY_AUTH_HTPASSWD_REALM
          value: basic_realm
        - name: REGISTRY_AUTH_HTPASSWD_PATH
          value: /auth/htpasswd
        volumeMounts:
        - name: image-store
          mountPath: /var/lib/registry
        - name: auth-dir
          mountPath: /auth
        ports:
        - containerPort: 5000
          name: registry
          protocol: TCP
      volumes:
      - name: image-store
        emptyDir: {}
      - name: auth-dir
        secret:
          secretName: registry-auth-secret
```
<!-- END MUNGE: EXAMPLE registry-auth-rc.yaml -->

No changes are needed for other components (kube-registry service and proxy). 

### To Verify

Setup proxy or port-forwarding to the kube-registry. Image push/pull should fail without authentication. Then use `docker login` to authenticate with kube-registry and see if it works.

### Configure Nodes to Authenticate with Kube-Registry

By default, nodes assume no authentication is required by kube-registry. Without authentication, nodes cannot pull images from kube-registry. To solve this, more documentation can be found [Here](https://github.com/kubernetes/kubernetes.github.io/blob/master/docs/concepts/containers/images.md#configuring-nodes-to-authenticate-to-a-private-repository).





[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/registry/auth/README.md?pixel)]()
