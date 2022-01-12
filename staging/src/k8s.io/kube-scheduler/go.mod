// This is a generated file. Do not edit directly.

module k8s.io/kube-scheduler

go 1.16

require (
	github.com/google/go-cmp v0.5.5
	k8s.io/api v0.23.0
	k8s.io/apimachinery v0.23.0
	k8s.io/component-base v0.23.0
	sigs.k8s.io/yaml v1.2.0
)

replace (
	github.com/hashicorp/golang-lru => github.com/hashicorp/golang-lru v0.5.0
	github.com/imdario/mergo => github.com/imdario/mergo v0.3.5
	github.com/onsi/ginkgo => github.com/openshift/ginkgo v4.7.0-origin.0+incompatible
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/component-base => ../component-base
	k8s.io/kube-scheduler => ../kube-scheduler
	sigs.k8s.io/structured-merge-diff/v4 => sigs.k8s.io/structured-merge-diff/v4 v4.1.2
)
