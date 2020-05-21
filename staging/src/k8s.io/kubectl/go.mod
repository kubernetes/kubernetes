// This is a generated file. Do not edit directly.

module k8s.io/kubectl

go 1.13

require (
	github.com/MakeNowJust/heredoc v0.0.0-20170808103936-bb23615498cd
	github.com/chai2010/gettext-go v0.0.0-20160711120539-c6fed771bfd5
	github.com/davecgh/go-spew v1.1.1
	github.com/daviddengcn/go-colortext v0.0.0-20160507010035-511bcaf42ccd
	github.com/docker/distribution v2.7.1+incompatible
	github.com/docker/docker v0.7.3-0.20190327010347-be7ac8be2ae0
	github.com/evanphx/json-patch v4.2.0+incompatible
	github.com/exponent-io/jsonpath v0.0.0-20151013193312-d6023ce2651d
	github.com/fatih/camelcase v1.0.0
	github.com/go-openapi/spec v0.19.3
	github.com/google/go-cmp v0.3.1
	github.com/googleapis/gnostic v0.2.0
	github.com/jonboulle/clockwork v0.1.0
	github.com/liggitt/tabwriter v0.0.0-20181228230101-89fcab3d43de
	github.com/lithammer/dedent v1.1.0
	github.com/mitchellh/go-wordwrap v1.0.0
	github.com/onsi/ginkgo v1.11.0
	github.com/onsi/gomega v1.7.0
	github.com/russross/blackfriday v1.5.2
	github.com/spf13/cobra v0.0.5
	github.com/spf13/pflag v1.0.5
	github.com/stretchr/testify v1.4.0
	golang.org/x/sys v0.0.0-20191022100944-742c48ecaeb7
	gopkg.in/yaml.v2 v2.2.8
	k8s.io/api v0.18.3
	k8s.io/apimachinery v0.18.3
	k8s.io/cli-runtime v0.0.0
	k8s.io/client-go v0.18.3
	k8s.io/component-base v0.18.3
	k8s.io/klog v1.0.0
	k8s.io/kube-openapi v0.0.0-20200410145947-61e04a5be9a6 // release-1.18
	k8s.io/metrics v0.0.0
	k8s.io/utils v0.0.0-20200324210504-a9aa75ae1b89
	sigs.k8s.io/kustomize v2.0.3+incompatible
	sigs.k8s.io/yaml v1.2.0
	vbom.ml/util v0.0.0-20180919145318-efcd4e0f9787
)

replace (
	github.com/coreos/etcd => github.com/coreos/etcd v3.3.10+incompatible
	github.com/go-bindata/go-bindata => github.com/go-bindata/go-bindata v3.1.1+incompatible
	github.com/google/go-cmp => github.com/google/go-cmp v0.3.0
	github.com/googleapis/gnostic => github.com/googleapis/gnostic v0.1.0
	github.com/imdario/mergo => github.com/imdario/mergo v0.3.5
	github.com/json-iterator/go => github.com/json-iterator/go v1.1.8
	github.com/prometheus/client_golang => github.com/prometheus/client_golang v1.0.0
	github.com/vishvananda/netns => github.com/vishvananda/netns v0.0.0-20171111001504-be1fbeda1936
	go.uber.org/atomic => go.uber.org/atomic v1.3.2
	go.uber.org/multierr => go.uber.org/multierr v1.1.0
	golang.org/x/net => golang.org/x/net v0.0.0-20191004110552-13f9640d40b9
	golang.org/x/sys => golang.org/x/sys v0.0.0-20190813064441-fde4db37ae7a // pinned to release-branch.go1.13
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190821162956-65e3620a7ae7 // pinned to release-branch.go1.13
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
