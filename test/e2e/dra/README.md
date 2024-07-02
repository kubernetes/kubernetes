# Overview

The tests in this directory cover dynamic resource allocation support in
Kubernetes. They do not test the correct behavior of arbitrary dynamic resource
allocation drivers.

If such a driver is needed, then the in-tree test/e2e/dra/test-driver is used,
with a slight twist: instead of deploying that driver directly in the cluster,
the necessary sockets for interaction with kubelet (registration and dynamic
resource allocation) get proxied into the e2e.test binary. This reuses the work
done for CSI mock testing. The advantage is that no separate images are needed
for the test driver and that the e2e test has full control over all gRPC calls,
in case that it needs that for operations like error injection or checking
calls.

## Cluster setup preparation

The container runtime must support CDI. CRI-O supports CDI starting from release 1.23,
Containerd supports CDI starting from release 1.7. To bring up a Kind cluster with Containerd,
two things are needed:
- [build binaries from Kubernetes source code tree](https://github.com/kubernetes/community/blob/master/contributors/devel/development.md#building-kubernetes)
- [Kind](https://github.com/kubernetes-sigs/kind)

> NB: Kind switched to use worker-node base image with Containerd 1.7 by default starting from
release 0.20, build kind from latest main branch sources or use Kind release binary 0.20 or later.

### Build kind node image

After building Kubernetes, in Kubernetes source code tree build new node image:
```bash
$ kind build node-image --image dra/node:latest $(pwd)
```

## Bring up a Kind cluster
```bash
$ kind create cluster --config test/e2e/dra/kind.yaml --image dra/node:latest
```

### With support for cluster autoscaling

A workload cluster set up with [Cluster API](https://cluster-api.sigs.k8s.io)
(CAPI) based on the Docker provider supports autoscaling. CAPI runs inside a
Kind cluster. It uses the kind node image for the workload cluster but talks
directly to Docker and kubeadm to create and join nodes.

Setting it up is roughly as described in
[quickstart](https://cluster-api.sigs.k8s.io/user/quick-start), but one has to
be careful to use the instructions for the Docker provider. The following steps
build Kubernetes from scratch and enable DRA:

```bash
$ version=$( . hack/lib/version.sh; KUBE_ROOT=`pwd`; kube::version::get_version_vars; echo $KUBE_GIT_VERSION )
$ echo $version
v1.30.0-alpha.0.47+1f1e7f781cf3aa-dirty


$ kind build node-image --image kindest/node:${version//+/_} $(pwd)

# CAPI needs access to Docker.
$ cat >kind-cluster-with-extramounts.yaml <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
networking:
  ipFamily: dual
nodes:
- role: control-plane
  extraMounts:
    - hostPath: /var/run/docker.sock
      containerPath: /var/run/docker.sock
EOF

$ kind create cluster --config kind-cluster-with-extramounts.yaml
...
Creating cluster "kind" ...
...

$ CLUSTER_TOPOLOGY=true clusterctl init --infrastructure docker
Fetching providers
Installing cert-manager Version="v1.13.2"
Waiting for cert-manager to be available...
Installing Provider="cluster-api" Version="v1.6.0" TargetNamespace="capi-system"
Installing Provider="bootstrap-kubeadm" Version="v1.6.0" TargetNamespace="capi-kubeadm-bootstrap-system"
Installing Provider="control-plane-kubeadm" Version="v1.6.0" TargetNamespace="capi-kubeadm-control-plane-system"
Installing Provider="infrastructure-docker" Version="v1.6.0" TargetNamespace="capd-system"

Your management cluster has been initialized successfully!

You can now create your first workload cluster by running the following:

  clusterctl generate cluster [name] --kubernetes-version [version] | kubectl apply -f -

# Patching in the configuration for pod security conflicts with
# extraArgs for apisever due to this JSON patch:
#     - jsonPatches:
#      - op: add
#        path: /spec/template/spec/kubeadmConfigSpec/clusterConfiguration/apiServer/extraArgs
#        value:
#         admission-control-config-file: /etc/kubernetes/kube-apiserver-admission-pss.yaml

$ POD_SECURITY_STANDARD_ENABLED=false clusterctl generate cluster capi-quickstart --flavor development   --kubernetes-version $version --control-plane-machine-count=1   --worker-machine-count=1 >cluster-latest.yaml

# MachinePools are experimental, so this patch removes them from the cluster.
# Autoscaling will be done through MachineDeployments.

$ cat cluster.patch <<EOF
*** cluster-latest.yaml.orig	2024-03-25 16:30:22.172210145 +0100
--- cluster-latest.yaml	2024-03-25 16:33:15.645456367 +0100
***************
*** 250,255 ****
--- 250,258 ----
        kubeadmConfigSpec:
          clusterConfiguration:
            apiServer:
+             extraArgs:
+               runtime-config: resource.k8s.io/v1alpha2=true
+               feature-gates: DynamicResourceAllocation=true,ContextualLogging=true
              certSANs:
              - localhost
              - 127.0.0.1
***************
*** 258,263 ****
--- 261,270 ----
            controllerManager:
              extraArgs:
                enable-hostpath-provisioner: "true"
+               feature-gates: DynamicResourceAllocation=true,ContextualLogging=true
+           scheduler:
+             extraArgs:
+               feature-gates: DynamicResourceAllocation=true,ContextualLogging=true
          initConfiguration:
            nodeRegistration: {}
          joinConfiguration:
***************
*** 306,312 ****
    template:
      spec:
        joinConfiguration:
!         nodeRegistration: {}
  ---
  apiVersion: cluster.x-k8s.io/v1beta1
  kind: Cluster
--- 313,325 ----
    template:
      spec:
        joinConfiguration:
!         nodeRegistration:
!           kubeletExtraArgs:
!             feature-gates: DynamicResourceAllocation=true,ContextualLogging=true
!       initConfiguration:
!         nodeRegistration:
!           kubeletExtraArgs:
!             feature-gates: DynamicResourceAllocation=true,ContextualLogging=true
  ---
  apiVersion: cluster.x-k8s.io/v1beta1
  kind: Cluster
***************
*** 346,352 ****
        - class: default-worker
          name: md-0
          replicas: 1
-       machinePools:
-       - class: default-worker
-         name: mp-0
-         replicas: 1
--- 359,361 ----
EOF

# patch cluster-latest.yaml <cluster.patch
patching file cluster-latest.yaml

$ kubectl apply -f cluster-latest.yaml
clusterclass.cluster.x-k8s.io/quick-start created
dockerclustertemplate.infrastructure.cluster.x-k8s.io/quick-start-cluster created
kubeadmcontrolplanetemplate.controlplane.cluster.x-k8s.io/quick-start-control-plane created
dockermachinetemplate.infrastructure.cluster.x-k8s.io/quick-start-control-plane created
dockermachinetemplate.infrastructure.cluster.x-k8s.io/quick-start-default-worker-machinetemplate created
dockermachinepooltemplate.infrastructure.cluster.x-k8s.io/quick-start-default-worker-machinepooltemplate created
kubeadmconfigtemplate.bootstrap.cluster.x-k8s.io/quick-start-default-worker-bootstraptemplate created
cluster.cluster.x-k8s.io/capi-quickstart created

# Some time later (minutes, not hours)...

$ kubectl get cluster
NAME              CLUSTERCLASS   PHASE         AGE   VERSION
capi-quickstart   quick-start    Provisioned   47s   v1.29.0
$ kubectl get machine
NAME                                     CLUSTER           NODENAME                                 PROVIDERID                                          PHASE     AGE   VERSION
capi-quickstart-h5bc8-wgppr              capi-quickstart   capi-quickstart-h5bc8-wgppr              docker:////capi-quickstart-h5bc8-wgppr              Running   89s   v1.29.0
capi-quickstart-md-0-tzjwj-tb92p-wt4dc   capi-quickstart   capi-quickstart-md-0-tzjwj-tb92p-wt4dc   docker:////capi-quickstart-md-0-tzjwj-tb92p-wt4dc   Running   91s   v1.29.0

$ clusterctl get kubeconfig capi-quickstart > capi-quickstart.kubeconfig

$ KUBECONFIG=capi-quickstart.kubeconfig kubectl get nodes
NAME                                     STATUS     ROLES           AGE     VERSION
capi-quickstart-h5bc8-wgppr              NotReady   control-plane   2m26s   v1.29.0
capi-quickstart-md-0-tzjwj-tb92p-wt4dc   NotReady   <none>          2m8s    v1.29.0

# The control plane won’t be Ready until we install a CNI in the next step.

$ KUBECONFIG=capi-quickstart.kubeconfig kubectl --kubeconfig=./capi-quickstart.kubeconfig \
  apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.1/manifests/calico.yaml
...
```

The cluster autoscaler then can be run outside of any cluster with:

```
cluster-autoscaler$ go run . - -v3 --cloud-provider=clusterapi --cloud-config=$HOME/.kube/config --kubeconfig=<path to Kubernetes>/capi-quickstart.kubeconfig
```

## Run tests

- Build ginkgo

```bash
$ make ginkgo
```

- Run e2e tests for the `Dynamic Resource Allocation` feature:

```bash
$ KUBECONFIG=~/.kube/config _output/bin/ginkgo -p -v -focus=Feature:DynamicResourceAllocation ./test/e2e
```
