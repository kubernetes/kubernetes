// This is a generated file. Do not edit directly.

module k8s.io/sample-cli-plugin

go 1.15

require (
	github.com/emicklei/go-restful v2.9.5+incompatible // indirect
	github.com/mailru/easyjson v0.7.0 // indirect
	github.com/spf13/cobra v1.1.1
	github.com/spf13/pflag v1.0.5
	k8s.io/cli-runtime v0.0.0
	k8s.io/client-go v0.20.0
)

replace (
	github.com/imdario/mergo => github.com/imdario/mergo v0.3.5
	github.com/onsi/ginkgo => github.com/openshift/ginkgo v4.5.0-origin.1+incompatible
	gopkg.in/yaml.v2 => gopkg.in/yaml.v2 v2.2.8
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/cli-runtime => ../cli-runtime
	k8s.io/client-go => ../client-go
	k8s.io/sample-cli-plugin => ../sample-cli-plugin
)
