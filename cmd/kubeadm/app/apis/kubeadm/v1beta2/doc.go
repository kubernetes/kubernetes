/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// +k8s:defaulter-gen=TypeMeta
// +groupName=kubeadm.k8s.io
// +k8s:deepcopy-gen=package
// +k8s:conversion-gen=k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm

// Package v1beta2 defines the v1beta2 version of the kubeadm configuration file format.
// This version improves on the v1beta1 format by fixing some minor issues and adding a few new fields.
//
// A list of changes since v1beta1:
//
// - `certificateKey` field is added to `InitConfiguration` and `JoinConfiguration`.
// - `ignorePreflightErrors` field is added to the `NodeRegistrationOptions`.
// - The JSON `omitempty` tag is used in more places where appropriate.
// - The JSON `omitempty` tag of the "taints" field (in `NodeRegistrationOptions`) is removed.
//
// See the Kubernetes 1.15 changelog for further details.
//
// **Migration from old kubeadm config versions**
//
// Please convert your v1beta1 configuration files to v1beta2 using the `kubeadm config migrate` command of kubeadm
// v1.15.x. kubeadm v1.15.x supports reading from v1beta1 version of the kubeadm config file format.
//
// **Basics**
//
// The preferred way to configure kubeadm is to pass an YAML configuration file with the `--config` option. Some of the
// configuration options defined in the kubeadm config file are also available as command line flags, but only
// the most common/simple use case are supported with this approach.
//
// A kubeadm config file could contain multiple configuration types separated using three dashes (`---`).
//
// kubeadm supports the following configuration types:
//
// ```yaml
// apiVersion: kubeadm.k8s.io/v1beta2
// kind: InitConfiguration
// ```
// ```yaml
// apiVersion: kubeadm.k8s.io/v1beta2
// kind: ClusterConfiguration
// ```
// ```yaml
// apiVersion: kubelet.config.k8s.io/v1beta1
// kind: KubeletConfiguration
// ```
// ```yaml
// apiVersion: kubeproxy.config.k8s.io/v1alpha1
// kind: KubeProxyConfiguration
// ```
// ```yaml
// apiVersion: kubeadm.k8s.io/v1beta2
// kind: JoinConfiguration
// ```
//
// To print the defaults for `init` and `join` actions use the following commands:
//
// ```shell
// kubeadm config print init-defaults
// kubeadm config print join-defaults
// ```
//
// The list of configuration types that must be included in a configuration file depends by the action you are
// performing (`init` or `join`) and by the configuration options you are going to use (defaults or advanced customization).
//
// If some configuration types are not provided, or provided only partially, kubeadm will use default values.
// Defaults provided by kubeadm help enforce consistency of values across components when required (e.g.
// `--cluster-cidr` flag on controller manager and `clusterCIDR` on kube-proxy).
//
// Users are always allowed to override default values, with the only exception of a small subset of setting
// related to security (e.g. enforce authorization-mode Node and RBAC on the API server)
// If the user provides a configuration types that is not expected for the action you are performing, kubeadm will
// ignore those types and print a warning.
//
// **Kubeadm init configuration types**
//
// When executing kubeadm init with the `--config` option, the following configuration types could be used:
// `InitConfiguration`, `ClusterConfiguration`, `KubeProxyConfiguration`, `KubeletConfiguration`, but only one
// between `InitConfiguration` and `ClusterConfiguration` is mandatory.
//
// ```yaml
// apiVersion: kubeadm.k8s.io/v1beta2
// kind: InitConfiguration
// bootstrapTokens:
//   # ...
// nodeRegistration:
//   # ...
// ```
//
// The `InitConfiguration` type is used to configure runtime settings. In the case of `kubeadm init`,
// it includes the configuration of the bootstrap token and all the setting specific to the node
// where kubeadm is executed, including:
//
// - `nodeRegistration`: fields that relate to registering the new node to the cluster.
//   You can use it to customize the node name, the CRI socket to use or any other
//   settings that should apply to this node only (for example. the node IP).
//
// - `localAPIEndpoint`: the endpoint of the API server instance to be deployed on this node.
//   For example, you can use it to customize the API server advertise address.
//
//   ```yaml
//   apiVersion: kubeadm.k8s.io/v1beta2
//   kind: ClusterConfiguration
//   networking:
//     # ...
//   etcd:
//     # ...
//   apiServer:
//     extraArgs:
//       # ...
//     extraVolumes:
//       # ...
//   # ...
//   ```
//
// The `ClusterConfiguration` type can be used to configure cluster-wide settings, including:
//
// - Networking, that holds configuration for the networking topology of the cluster;
//   For example, uou can use it to customize node subnet or services subnet.
//
// - Etcd configurations that can be used for customizing the local etcd or to configure
//   the API server for using an external etcd cluster.
//
// - kube-apiserver, kube-scheduler, kube-controller-manager configurations.
//   You can use it to customize control-plane components by adding customized setting
//   or overriding kubeadm default settings.
//
// ```yaml
// apiVersion: kubeproxy.config.k8s.io/v1alpha1
// kind: KubeProxyConfiguration
// # ...
// ```
//
// The `KubeProxyConfiguration` type is used to change the configurations passed to the kube-proxy
// instances deployed in the cluster. If this object is not provided or provided only partially,
// kubeadm applies defaults.
// See [kube-proxy reference](https://kubernetes.io/docs/reference/command-line-tools-reference/kube-proxy/)
// or [kube-proxy GoDoc](https://godoc.org/k8s.io/kube-proxy/config/v1alpha1#KubeProxyConfiguration)
// for the official documentation.
//
// ```yaml
// apiVersion: kubelet.config.k8s.io/v1beta1
// kind: KubeletConfiguration
// # ...
// ```
//
// The `KubeletConfiguration` type is used to change the configurations passed to all kubelet instances
// deployed in the cluster. If this object is not provided or provided only partially, kubeadm applies defaults.
// See [kubelet reference](https://kubernetes.io/docs/reference/command-line-tools-reference/kubelet/) or
// [kubelet GoDoc](https://godoc.org/k8s.io/kubelet/config/v1beta1#KubeletConfiguration)
// for the official documentation.
//
// Here is a fully populated example of a single YAML file containing multiple
// configuration types to be used during a `kubeadm init` run.
//
// ```yaml
// apiVersion: kubeadm.k8s.io/v1beta2
// kind: InitConfiguration
// bootstrapTokens:
// - token: "9a08jv.c0izixklcxtmnze7"
//   description: "kubeadm bootstrap token"
//   ttl: "24h"
// - token: "783bde.3f89s0fje9f38fhf"
//   description: "another bootstrap token"
//   usages:
//   - authentication
//   - signing
//   groups:
//   - system:bootstrappers:kubeadm:default-node-token
// nodeRegistration:
//   name: "ec2-10-100-0-1"
//   criSocket: "/var/run/dockershim.sock"
//   taints:
//   - key: "kubeadmNode"
//     value: "master"
//     effect: "NoSchedule"
//   kubeletExtraArgs:
//     cgroup-driver: "cgroupfs"
//   ignorePreflightErrors:
//   - IsPrivilegedUser
// localAPIEndpoint:
//   advertiseAddress: "10.100.0.1"
//   bindPort: 6443
// certificateKey: "e6a2eb8581237ab72a4f494f30285ec12a9694d750b9785706a83bfcbbbd2204"
// ---
// apiVersion: kubeadm.k8s.io/v1beta2
// kind: ClusterConfiguration
// etcd:
//   # one of local or external
//   local:
//     imageRepository: "k8s.gcr.io"
//     imageTag: "3.2.24"
//     dataDir: "/var/lib/etcd"
//     extraArgs:
//       listen-client-urls: "http://10.100.0.1:2379"
//     serverCertSANs:
//     -  "ec2-10-100-0-1.compute-1.amazonaws.com"
//     peerCertSANs:
//     - "10.100.0.1"
//   # external:
//     # endpoints:
//     # - "10.100.0.1:2379"
//     # - "10.100.0.2:2379"
//     # caFile: "/etcd/kubernetes/pki/etcd/etcd-ca.crt"
//     # certFile: "/etcd/kubernetes/pki/etcd/etcd.crt"
//     # keyFile: "/etcd/kubernetes/pki/etcd/etcd.key"
// networking:
//   serviceSubnet: "10.96.0.0/12"
//   podSubnet: "10.100.0.1/24"
//   dnsDomain: "cluster.local"
// kubernetesVersion: "v1.12.0"
// controlPlaneEndpoint: "10.100.0.1:6443"
// apiServer:
//   extraArgs:
//     authorization-mode: "Node,RBAC"
//   extraVolumes:
//   - name: "some-volume"
//     hostPath: "/etc/some-path"
//     mountPath: "/etc/some-pod-path"
//     readOnly: false
//     pathType: File
//   certSANs:
//   - "10.100.1.1"
//   - "ec2-10-100-0-1.compute-1.amazonaws.com"
//   timeoutForControlPlane: 4m0s
// controllerManager:
//   extraArgs:
//     "node-cidr-mask-size": "20"
//   extraVolumes:
//   - name: "some-volume"
//     hostPath: "/etc/some-path"
//     mountPath: "/etc/some-pod-path"
//     readOnly: false
//     pathType: File
// scheduler:
//   extraArgs:
//     address: "10.100.0.1"
//   extraVolumes:
//   - name: "some-volume"
//     hostPath: "/etc/some-path"
//     mountPath: "/etc/some-pod-path"
//     readOnly: false
//     pathType: File
// certificatesDir: "/etc/kubernetes/pki"
// imageRepository: "k8s.gcr.io"
// useHyperKubeImage: false
// clusterName: "example-cluster"
// ---
// apiVersion: kubelet.config.k8s.io/v1beta1
// kind: KubeletConfiguration
// # kubelet specific options here
// ---
// apiVersion: kubeproxy.config.k8s.io/v1alpha1
// kind: KubeProxyConfiguration
// # kube-proxy specific options here
// ```
//
// **Kubeadm join configuration types**
//
// When executing `kubeadm join` with the `--config` option, the `JoinConfiguration` type should be provided.
//
// ```yaml
// apiVersion: kubeadm.k8s.io/v1beta2
// kind: JoinConfiguration
// # ...
// ```
//
// The `JoinConfiguration` type is used to configure runtime settings. In the case of `kubeadm join`,
// it contains the discovery method used for accessing the cluster info and all the setting which are specific
// to the node where kubeadm is executed, including:
//
// - `nodeRegistration`: fields related to registering the new node to the cluster.
//   You can use it to customize the node name, the CRI socket to use or any other settings that should
//   apply to this node only (e.g. the node ip).
//
// - `apiEndpoint`: the endpoint of the API server instance to be eventually deployed on this node.
//
package v1beta2 // import "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"

//TODO: The BootstrapTokenString object should move out to either k8s.io/client-go or k8s.io/api in the future
//(probably as part of Bootstrap Tokens going GA). It should not be staged under the kubeadm API as it is now.
