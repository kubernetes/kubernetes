// This is a generated file. Do not edit directly.

module k8s.io/cloud-provider

go 1.13

require (
	github.com/google/go-cmp v0.4.0
	github.com/stretchr/testify v1.4.0
	k8s.io/api v0.19.0-rc.2
	k8s.io/apimachinery v0.19.0-rc.2
	k8s.io/client-go v0.19.0-rc.2
	k8s.io/component-base v0.19.0-rc.2
	k8s.io/klog/v2 v2.2.0
	k8s.io/utils v0.0.0-20200720150651-0bdb4ca86cbc
)

replace (
	github.com/containerd/continuity => github.com/containerd/continuity v0.0.0-20190426062206-aaeac12a7ffc
	github.com/coreos/etcd => github.com/coreos/etcd v3.3.10+incompatible
	github.com/evanphx/json-patch => github.com/evanphx/json-patch v0.0.0-20190815234213-e83c0a1c26c8
	github.com/go-bindata/go-bindata => github.com/go-bindata/go-bindata v3.1.1+incompatible
	github.com/golang/glog => github.com/openshift/golang-glog v0.0.0-20190322123450-3c92600d7533
	github.com/imdario/mergo => github.com/imdario/mergo v0.3.5
	github.com/openshift/build-machinery-go => github.com/openshift/build-machinery-go v0.0.0-20200424080330-082bf86082cc
	github.com/robfig/cron => github.com/robfig/cron v1.1.0
	go.uber.org/multierr => go.uber.org/multierr v1.1.0
	golang.org/x/net => golang.org/x/net v0.0.0-20200324143707-d3edc9973b7e
	gopkg.in/yaml.v2 => gopkg.in/yaml.v2 v2.2.8
	k8s.io/api => ../api
	k8s.io/apiextensions-apiserver => ../apiextensions-apiserver
	k8s.io/apimachinery => ../apimachinery
	k8s.io/apiserver => ../apiserver
	k8s.io/cli-runtime => ../cli-runtime
	k8s.io/client-go => ../client-go
	k8s.io/cloud-provider => ../cloud-provider
	k8s.io/cluster-bootstrap => ../cluster-bootstrap
	k8s.io/code-generator => ../code-generator
	k8s.io/component-base => ../component-base
	k8s.io/cri-api => ../cri-api
	k8s.io/csi-translation-lib => ../csi-translation-lib
	k8s.io/kube-aggregator => ../kube-aggregator
	k8s.io/kube-controller-manager => ../kube-controller-manager
	k8s.io/kube-proxy => ../kube-proxy
	k8s.io/kube-scheduler => ../kube-scheduler
	k8s.io/kubectl => ../kubectl
	k8s.io/kubelet => ../kubelet
	k8s.io/legacy-cloud-providers => ../legacy-cloud-providers
	k8s.io/metrics => ../metrics
	k8s.io/sample-apiserver => ../sample-apiserver
	vbom.ml/util => vbom.ml/util v0.0.0-20160121211510-db5cfe13f5cc
)
