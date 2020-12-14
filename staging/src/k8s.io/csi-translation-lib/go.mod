// This is a generated file. Do not edit directly.

module k8s.io/csi-translation-lib

go 1.15

require (
	github.com/stretchr/testify v1.6.1
	k8s.io/api v0.20.0
	k8s.io/apimachinery v0.20.0
	k8s.io/klog/v2 v2.4.0
)

replace (
	github.com/onsi/ginkgo => github.com/openshift/ginkgo v4.5.0-origin.1+incompatible
	gopkg.in/yaml.v2 => gopkg.in/yaml.v2 v2.2.8
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/csi-translation-lib => ../csi-translation-lib
)
