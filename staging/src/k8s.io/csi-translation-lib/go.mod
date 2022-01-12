// This is a generated file. Do not edit directly.

module k8s.io/csi-translation-lib

go 1.16

require (
	github.com/stretchr/testify v1.7.0
	k8s.io/api v0.23.0
	k8s.io/apimachinery v0.23.0
	k8s.io/klog/v2 v2.30.0
)

replace (
	github.com/onsi/ginkgo => github.com/openshift/ginkgo v4.7.0-origin.0+incompatible
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/csi-translation-lib => ../csi-translation-lib
	sigs.k8s.io/structured-merge-diff/v4 => sigs.k8s.io/structured-merge-diff/v4 v4.1.2
)
