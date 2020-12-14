// This is a generated file. Do not edit directly.

module k8s.io/cluster-bootstrap

go 1.15

require (
	github.com/stretchr/testify v1.6.1
	golang.org/x/crypto v0.0.0-20201002170205-7f63de1d35b0 // indirect
	gopkg.in/square/go-jose.v2 v2.2.2
	k8s.io/api v0.20.0
	k8s.io/apimachinery v0.20.0
	k8s.io/klog/v2 v2.4.0
)

replace (
	github.com/onsi/ginkgo => github.com/openshift/ginkgo v4.5.0-origin.1+incompatible
	gopkg.in/yaml.v2 => gopkg.in/yaml.v2 v2.2.8
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/cluster-bootstrap => ../cluster-bootstrap
)
