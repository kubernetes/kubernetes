// This is a generated file. Do not edit directly.

module k8s.io/controller-manager

go 1.16

require (
	github.com/spf13/pflag v1.0.5
	github.com/stretchr/testify v1.7.0
	golang.org/x/oauth2 v0.0.0-20210819190943-2bc19b11175f
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/apiserver v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/component-base v0.0.0
	k8s.io/klog/v2 v2.30.0
	k8s.io/utils v0.0.0-20210930125809-cb0fa318a74b
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/apiserver => ../apiserver
	k8s.io/client-go => ../client-go
	k8s.io/component-base => ../component-base
	k8s.io/controller-manager => ../controller-manager
)
