// This is a generated file. Do not edit directly.

module k8s.io/kube-proxy

go 1.16

require (
	k8s.io/apimachinery v0.0.0
	k8s.io/component-base v0.0.0
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/component-base => ../component-base
	k8s.io/gengo => github.com/liggitt/gengo v0.0.0-20211011145608-112dde69eb87
	k8s.io/kube-proxy => ../kube-proxy
)
