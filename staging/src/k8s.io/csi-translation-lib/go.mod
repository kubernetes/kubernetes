// This is a generated file. Do not edit directly.

module k8s.io/csi-translation-lib

go 1.16

require (
	github.com/stretchr/testify v1.8.0
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/klog/v2 v2.30.0
)

replace (
	golang.org/x/sys => golang.org/x/sys v0.0.0-20220908164124-27713097b956
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/csi-translation-lib => ../csi-translation-lib
)
