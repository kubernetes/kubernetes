// This is a generated file. Do not edit directly.

module k8s.io/component-helpers

go 1.15

require (
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/klog/v2 v2.4.0
	k8s.io/utils v0.0.0-20201110183641-67b214c5f920
)

replace (
	golang.org/x/net => golang.org/x/net v0.0.0-20201110031124-69a78807bb2b
	golang.org/x/sys => golang.org/x/sys v0.0.0-20201112073958-5cba982894dd
	google.golang.org/protobuf => google.golang.org/protobuf v1.25.0
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/component-helpers => ../component-helpers
)
