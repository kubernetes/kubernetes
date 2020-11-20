// This is a generated file. Do not edit directly.

module k8s.io/kube-scheduler

go 1.15

require (
	github.com/google/go-cmp v0.5.2
	k8s.io/api v0.20.0
	k8s.io/apimachinery v0.20.0
	k8s.io/component-base v0.20.0
	sigs.k8s.io/yaml v1.2.0
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
	k8s.io/kube-scheduler => ../kube-scheduler
)
