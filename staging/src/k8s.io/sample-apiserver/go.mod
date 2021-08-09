// This is a generated file. Do not edit directly.

module k8s.io/sample-apiserver

go 1.15

require (
	github.com/go-openapi/spec v0.19.3
	github.com/google/gofuzz v1.1.0
	github.com/spf13/cobra v1.1.1
	k8s.io/apimachinery v0.0.0
	k8s.io/apiserver v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/code-generator v0.0.0
	k8s.io/component-base v0.0.0
	k8s.io/klog/v2 v2.4.0
	k8s.io/kube-openapi v0.0.0-20201113171705-d219536bb9fd
)

replace (
	github.com/sirupsen/logrus => github.com/sirupsen/logrus v1.6.0
	golang.org/x/net => golang.org/x/net v0.0.0-20201110031124-69a78807bb2b
	golang.org/x/sys => golang.org/x/sys v0.0.0-20201112073958-5cba982894dd
	google.golang.org/protobuf => google.golang.org/protobuf v1.25.0
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/apiserver => ../apiserver
	k8s.io/client-go => ../client-go
	k8s.io/code-generator => ../code-generator
	k8s.io/component-base => ../component-base
	k8s.io/sample-apiserver => ../sample-apiserver
)
