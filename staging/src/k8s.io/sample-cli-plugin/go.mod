// This is a generated file. Do not edit directly.

module k8s.io/sample-cli-plugin

go 1.16

require (
	github.com/spf13/cobra v1.2.1
	github.com/spf13/pflag v1.0.5
	k8s.io/cli-runtime v0.0.0
	k8s.io/client-go v0.0.0
)

replace (
	github.com/golang/mock => github.com/golang/mock v1.5.0
	github.com/onsi/ginkgo => github.com/onsi/ginkgo v1.14.0
	github.com/onsi/gomega => github.com/onsi/gomega v1.10.1
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/cli-runtime => ../cli-runtime
	k8s.io/client-go => ../client-go
	k8s.io/sample-cli-plugin => ../sample-cli-plugin
)
