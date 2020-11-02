// This is a generated file. Do not edit directly.

module k8s.io/csi-translation-lib

go 1.15

require (
	github.com/stretchr/testify v1.4.0
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/klog/v2 v2.2.0
)

replace (
	golang.org/x/sys => golang.org/x/sys v0.0.0-20200930185726-fdedc70b468f
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/csi-translation-lib => ../csi-translation-lib
)
