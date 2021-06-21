// This is a generated file. Do not edit directly.
// Ensure you've carefully read
// https://git.k8s.io/community/contributors/devel/sig-architecture/vendor.md
// Run hack/pin-dependency.sh to change pinned dependency versions.
// Run hack/update-vendor.sh to update go.mod files and the vendor directory.

module k8s.io/component-base

go 1.16

require (
	github.com/blang/semver v3.5.1+incompatible
	github.com/go-logr/logr v0.4.0
	github.com/google/go-cmp v0.5.5
	github.com/moby/term v0.0.0-20210610120745-9d4ed1856297
	github.com/prometheus/client_golang v1.11.0
	github.com/prometheus/client_model v0.2.0
	github.com/prometheus/common v0.26.0
	github.com/prometheus/procfs v0.6.0
	github.com/spf13/cobra v1.1.3
	github.com/spf13/pflag v1.0.5
	github.com/stretchr/testify v1.7.0
	go.uber.org/zap v1.17.0
	golang.org/x/sys v0.0.0-20210616094352-59db8d763f22
	k8s.io/apimachinery v0.20.6
	k8s.io/client-go v0.20.6
	k8s.io/klog/v2 v2.9.0
	k8s.io/utils v0.0.0-20210521133846-da695404a2bc
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/apiserver => ../apiserver
	k8s.io/client-go => ../client-go
	k8s.io/component-base => ../component-base
	k8s.io/cri-api => ../cri-api
)
