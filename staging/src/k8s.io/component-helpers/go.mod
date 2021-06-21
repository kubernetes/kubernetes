// This is a generated file. Do not edit directly.
// Ensure you've carefully read
// https://git.k8s.io/community/contributors/devel/sig-architecture/vendor.md
// Run hack/pin-dependency.sh to change pinned dependency versions.
// Run hack/update-vendor.sh to update go.mod files and the vendor directory.

module k8s.io/component-helpers

go 1.16

require (
	github.com/google/go-cmp v0.5.5
	google.golang.org/appengine v1.6.5 // indirect
	k8s.io/api v0.20.6
	k8s.io/apimachinery v0.20.6
	k8s.io/client-go v0.20.6
	k8s.io/klog/v2 v2.9.0
	k8s.io/utils v0.0.0-20210521133846-da695404a2bc
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/component-helpers => ../component-helpers
)
