// This is a generated file. Do not edit directly.

module k8s.io/sample-cli-plugin

go 1.16

require (
	github.com/spf13/cobra v1.4.0
	github.com/spf13/pflag v1.0.5
	k8s.io/cli-runtime v0.0.0
	k8s.io/client-go v0.0.0
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/cli-runtime => ../cli-runtime
	k8s.io/client-go => ../client-go
	k8s.io/kube-openapi => github.com/jefftree/kube-openapi v0.0.8-gnostic.0.20220328150837-013580d8b582
	k8s.io/sample-cli-plugin => ../sample-cli-plugin
)
