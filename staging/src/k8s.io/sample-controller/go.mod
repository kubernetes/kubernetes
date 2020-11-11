// This is a generated file. Do not edit directly.

module k8s.io/sample-controller

go 1.15

require (
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/code-generator v0.0.0
	k8s.io/klog/v2 v2.4.0
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/code-generator => ../code-generator
	k8s.io/sample-controller => ../sample-controller
	sigs.k8s.io/structured-merge-diff/v4 => github.com/kwiesmueller/structured-merge-diff/v4 v4.0.0-20201110160604-4a28bb34fff5
)
