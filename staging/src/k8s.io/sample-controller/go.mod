// This is a generated file. Do not edit directly.

module k8s.io/sample-controller

go 1.12

require (
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/code-generator v0.0.0
	k8s.io/klog v1.0.0
)

replace (
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190821162956-65e3620a7ae7
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/code-generator => ../code-generator
	k8s.io/sample-controller => ../sample-controller
)
