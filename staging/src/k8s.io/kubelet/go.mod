// This is a generated file. Do not edit directly.

module k8s.io/kubelet

go 1.15

require (
	github.com/gogo/protobuf v1.3.1
	golang.org/x/net v0.0.0-20200707034311-ab3426394381
	google.golang.org/grpc v1.27.0
	k8s.io/api v0.19.2
	k8s.io/apimachinery v0.19.2
	k8s.io/component-base v0.19.2
)

replace (
	github.com/imdario/mergo => github.com/imdario/mergo v0.3.5
	github.com/onsi/ginkgo => github.com/openshift/ginkgo v4.5.0-origin.1+incompatible
	go.uber.org/multierr => go.uber.org/multierr v1.1.0
	gopkg.in/yaml.v2 => gopkg.in/yaml.v2 v2.2.8
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/component-base => ../component-base
	k8s.io/klog/v2 => k8s.io/klog/v2 v2.2.0
	k8s.io/kubelet => ../kubelet
)
