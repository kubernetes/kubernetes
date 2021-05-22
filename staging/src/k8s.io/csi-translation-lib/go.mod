// This is a generated file. Do not edit directly.

module k8s.io/csi-translation-lib

go 1.16

require (
	github.com/stretchr/testify v1.7.0
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/klog/v2 v2.8.0
)

replace (
	golang.org/x/net => golang.org/x/net v0.0.0-20210224082022-3d97a244fca7
	google.golang.org/grpc => google.golang.org/grpc v1.27.1
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/csi-translation-lib => ../csi-translation-lib
)
