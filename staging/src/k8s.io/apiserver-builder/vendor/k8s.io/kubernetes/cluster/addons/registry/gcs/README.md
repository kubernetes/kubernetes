# Kube-Registry with GCS storage backend

Besides local file system, docker registry also supports a number of cloud storage backends. Full list of supported backend can be found [here](https://docs.docker.com/registry/configuration/#storage). This document describes how to enable GCS for kube-registry as storage backend. 

A few preparation steps are needed. 
 1. Create a bucket named kube-registry in GCS.
 1. Create a service account for GCS access and create key file in json format. Detail instruction can be found [here](https://cloud.google.com/storage/docs/authentication#service_accounts).


### Pack Keyfile into a Secret

Assuming you have downloaded the keyfile as `keyfile.json`. Create secret with the `keyfile.json`...
```console
$ kubectl --namespace=kube-system create secret generic gcs-key-secret --from-file=keyfile=keyfile.json
```


### Run Registry

<!-- BEGIN MUNGE: EXAMPLE registry-gcs-rc.yaml -->
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
        - name: REGISTRY_STORAGE
          value: gcs
        - name: REGISTRY_STORAGE_GCS_BUCKET
          value: kube-registry
        - name: REGISTRY_STORAGE_GCS_KEYFILE
          value: /gcs/keyfile
        ports:
        - containerPort: 5000
          name: registry
          protocol: TCP
        volumeMounts:
        - name: gcs-key
          mountPath: /gcs
      volumes:
      - name: gcs-key
        secret:
          secretName: gcs-key-secret
```
<!-- END MUNGE: EXAMPLE registry-gcs-rc.yaml -->


No changes are needed for other components (kube-registry service and proxy). 


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/registry/gcs/README.md?pixel)]()
