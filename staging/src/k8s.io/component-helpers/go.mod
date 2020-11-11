// This is a generated file. Do not edit directly.

module k8s.io/component-helpers

go 1.15

require (
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/klog/v2 v2.4.0
	k8s.io/utils v0.0.0-20201104234853-8146046b121e
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/component-helpers => ../component-helpers
	sigs.k8s.io/structured-merge-diff/v4 => github.com/kwiesmueller/structured-merge-diff/v4 v4.0.0-20201110160604-4a28bb34fff5
)
