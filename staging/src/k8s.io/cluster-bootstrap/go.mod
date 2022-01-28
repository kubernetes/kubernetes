// This is a generated file. Do not edit directly.

module k8s.io/cluster-bootstrap

go 1.16

require (
	github.com/stretchr/testify v1.7.0
	golang.org/x/crypto v0.0.0-20210817164053-32db794688a5 // indirect
	gopkg.in/square/go-jose.v2 v2.2.2
	k8s.io/api v0.23.0
	k8s.io/apimachinery v0.23.0
	k8s.io/klog/v2 v2.30.0
)

replace (
	github.com/onsi/ginkgo => github.com/openshift/ginkgo v4.7.0-origin.0+incompatible
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/cluster-bootstrap => ../cluster-bootstrap
)
