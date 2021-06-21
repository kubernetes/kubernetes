// This is a generated file. Do not edit directly.
// Ensure you've carefully read
// https://git.k8s.io/community/contributors/devel/sig-architecture/vendor.md
// Run hack/pin-dependency.sh to change pinned dependency versions.
// Run hack/update-vendor.sh to update go.mod files and the vendor directory.

module k8s.io/sample-cli-plugin

go 1.16

require (
	github.com/spf13/cobra v1.1.3
	github.com/spf13/pflag v1.0.5
	k8s.io/cli-runtime v0.0.0
	k8s.io/client-go v0.20.6
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/cli-runtime => ../cli-runtime
	k8s.io/client-go => ../client-go
	k8s.io/sample-cli-plugin => ../sample-cli-plugin
)
