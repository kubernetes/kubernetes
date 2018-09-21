/*
Copyright 2018 The Kubernetes Authors.

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

// Package v1alpha3 is the API (config file) for driving the kubeadm binary.
// Some of these options are also available as command line flags, but
// the preferred way to configure kubeadm is to pass a single YAML file with
// multiple configuration types in with the --config option.
//
// kubeadm defines several configuration types:
// * InitConfiguration
// * JoinConfiguration
// * ClusterConfiguration
//
// InitConfiguration and JoinConfiguration cannot share a single YAML file,
// however it is expected that InitConfiguration and ClusterConfiguration will
// share a single YAML file.
//
//  A fully populated example of a single YAML file containing multiple
//  configuration types to be used during a `kubeadm init` run.
//	apiVersion: kubeadm.k8s.io/v1alpha3
//	kind: InitConfiguration
//	bootstrapTokens:
//	- token: "9a08jv.c0izixklcxtmnze7"
//	  description: "kubeadm bootstrap token"
//	  ttl: "24h"
//	- token: "783bde.3f89s0fje9f38fhf"
//	  description: "another bootstrap token"
//	  usages:
//	  - signing
//	  groups:
//	  - system:anonymous
//	nodeRegistration:
//	  name: "ec2-10-100-0-1"
//	  criSocket: "/var/run/dockershim.sock"
//	  taints:
//	  - key: "kubeadmNode"
//	    value: "master"
//	    effect: "NoSchedule"
//	  kubeletExtraArgs:
//	    cgroupDriver: "cgroupfs"
//	apiEndpoint:
//	  advertiseAddress: "10.100.0.1"
//	  bindPort: 6443
//	---
//	apiVersion: kubeadm.k8s.io/v1alpha3
//	kind: ClusterConfiguration
//	etcd:
//	  # one of local or external
//	  local:
//	    image: "k8s.gcr.io/etcd-amd64:3.2.18"
//	    dataDir: "/var/lib/etcd"
//	    extraArgs:
//	      listen-client-urls: "http://10.100.0.1:2379"
//	    serverCertSANs:
//	    -  "ec2-10-100-0-1.compute-1.amazonaws.com"
//	    peerCertSANs:
//	    - "10.100.0.1"
//	  external:
//	    endpoints:
//	    - "10.100.0.1:2379"
//	    - "10.100.0.2:2379"
//	    caFile: "/etcd/kubernetes/pki/etcd/etcd-ca.crt"
//	    certFile: "/etcd/kubernetes/pki/etcd/etcd.crt"
//	    certKey: "/etcd/kubernetes/pki/etcd/etcd.key"
//	networking:
//	  serviceSubnet: "10.96.0.0/12"
//	  podSubnet: "10.100.0.1/24"
//	  dnsDomain: "cluster.local"
//	kubernetesVersion: "v1.12.0"
//	controlPlaneEndpoint: "10.100.0.1:6443"
//	apiServerExtraArgs:
//	  authorization-mode: "Node,RBAC"
//	controlManagerExtraArgs:
//	  node-cidr-mask-size: 20
//	schedulerExtraArgs:
//	  address: "10.100.0.1"
//	apiServerExtraVolumes:
//	- name: "some-volume"
//	  hostPath: "/etc/some-path"
//	  mountPath: "/etc/some-pod-path"
//	  writable: true
//	  pathType: File
//	controllerManagerExtraVolumes:
//	- name: "some-volume"
//	  hostPath: "/etc/some-path"
//	  mountPath: "/etc/some-pod-path"
//	  writable: true
//	  pathType: File
//	schedulerExtraVolumes:
//	- name: "some-volume"
//	  hostPath: "/etc/some-path"
//	  mountPath: "/etc/some-pod-path"
//	  writable: true
//	  pathType: File
//	apiServerCertSANs:
//	- "10.100.1.1"
//	- "ec2-10-100-0-1.compute-1.amazonaws.com"
//	certificatesDir: "/etc/kubernetes/pki"
//	imageRepository: "k8s.gcr.io"
//	unifiedControlPlaneImage: "k8s.gcr.io/controlplane:v1.12.0"
//	auditPolicy:
//	  # https://kubernetes.io/docs/tasks/debug-application-cluster/audit/#audit-policy
//	  path: "/var/log/audit/audit.json"
//	  logDir: "/var/log/audit"
//	  logMaxAge: 7 # in days
//	featureGates:
//	  selfhosting: false
//	clusterName: "example-cluster"
//
// TODO: The BootstrapTokenString object should move out to either k8s.io/client-go or k8s.io/api in the future
// (probably as part of Bootstrap Tokens going GA). It should not be staged under the kubeadm API as it is now.
//
// +k8s:defaulter-gen=TypeMeta
// +groupName=kubeadm.k8s.io
// +k8s:deepcopy-gen=package
// +k8s:conversion-gen=k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm
package v1alpha3 // import "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
