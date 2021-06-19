// This is a generated file. Do not edit directly.

module k8s.io/component-helpers

go 1.16

require (
	github.com/google/go-cmp v0.5.4
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/klog/v2 v2.9.0
	k8s.io/utils v0.0.0-20210521133846-da695404a2bc
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/component-helpers => ../component-helpers
	k8s.io/kube-openapi => github.com/nikhita/kube-openapi v0.0.0-20210619182437-8f86f150b3e8
)
