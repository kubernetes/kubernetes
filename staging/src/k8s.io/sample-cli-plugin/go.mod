// This is a generated file. Do not edit directly.

module k8s.io/sample-cli-plugin

go 1.15

require (
	github.com/spf13/cobra v1.1.1
	github.com/spf13/pflag v1.0.5
	k8s.io/cli-runtime v0.0.0
	k8s.io/client-go v0.0.0
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/cli-runtime => ../cli-runtime
	k8s.io/client-go => ../client-go
	k8s.io/sample-cli-plugin => ../sample-cli-plugin
	sigs.k8s.io/structured-merge-diff/v4 => github.com/kwiesmueller/structured-merge-diff/v4 v4.0.0-20201110160604-4a28bb34fff5
)
