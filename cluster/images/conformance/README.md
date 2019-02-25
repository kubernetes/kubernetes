### conformance

`conformance` is a standalone container to launch Kubernetes end-to-end tests, for the purposes of conformance testing.
`conformance` is built for multiple architectures and _the image is pushed automatically on every release._

#### How to release by hand

```console
# First, build the binaries
$ build/run.sh make cross

# Build for linux/amd64 (default)
# export REGISTRY=$HOST/$ORG to switch from staging-k8s.gcr.io

$ make push VERSION={target_version} ARCH=amd64
# ---> staging-k8s.gcr.io/conformance-amd64:VERSION
# ---> staging-k8s.gcr.io/conformance:VERSION (image with backwards-compatible naming)

$ make push VERSION={target_version} ARCH=arm
# ---> staging-k8s.gcr.io/conformance-arm:VERSION

$ make push VERSION={target_version} ARCH=arm64
# ---> staging-k8s.gcr.io/conformance-arm64:VERSION

$ make push VERSION={target_version} ARCH=ppc64le
# ---> staging-k8s.gcr.io/conformance-ppc64le:VERSION

$ make push VERSION={target_version} ARCH=s390x
# ---> staging-k8s.gcr.io/conformance-s390x:VERSION
```

If you don't want to push the images, run `make` or `make build` instead


#### How to setup RBAC needed

```
kubectl create clusterrolebinding add-on-cluster-admin --clusterrole=cluster-admin --serviceaccount=default:default
```

#### How to run a single test

```
apiVersion: v1
kind: Pod
metadata:
  name: e2e-producer-consumer-test
spec:
  containers:
  - name: conformance-container
    image: gcr.io/heptio-images/kube-conformance:latest
    image: staging-k8s.gcr.io/conformance-amd64:v1.12.1
    imagePullPolicy: IfNotPresent
    env:
    - name: E2E_FOCUS
      value: "Pods should be submitted and removed"
    volumeMounts:
    - name: output-volume
      mountPath: /tmp/results
  volumes:
  - name: output-volume
    hostPath:
      path: /tmp/results
  restartPolicy: Never
```


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/images/conformance/README.md?pixel)]()
