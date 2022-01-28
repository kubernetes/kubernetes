// This is a generated file. Do not edit directly.

module k8s.io/component-helpers

go 1.16

require (
	github.com/google/go-cmp v0.5.5
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/klog/v2 v2.40.1
	k8s.io/utils v0.0.0-20211208161948-7d6a63dca704
)

replace (
	github.com/pquerna/cachecontrol => github.com/pquerna/cachecontrol v0.0.0-20171018203845-0dec1b30a021
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/apiserver => ../apiserver
	k8s.io/client-go => ../client-go
	k8s.io/component-base => ../component-base
	k8s.io/component-helpers => ../component-helpers
)
