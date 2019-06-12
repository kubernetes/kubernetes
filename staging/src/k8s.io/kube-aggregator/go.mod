// This is a generated file. Do not edit directly.

module k8s.io/kube-aggregator

go 1.12

require (
	github.com/davecgh/go-spew v1.1.1
	github.com/emicklei/go-restful v0.0.0-20170410110728-ff4f55a20633
	github.com/go-openapi/spec v0.17.2
	github.com/gogo/protobuf v0.0.0-20171007142547-342cbe0a0415
	github.com/inconshreveable/mousetrap v1.0.0 // indirect
	github.com/prometheus/client_golang v0.9.2
	github.com/spf13/cobra v0.0.0-20180319062004-c439c4fa0937
	github.com/spf13/pflag v1.0.1
	github.com/stretchr/testify v1.2.2
	golang.org/x/net v0.0.0-20190206173232-65e2d4e15006
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/apiserver v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/code-generator v0.0.0
	k8s.io/component-base v0.0.0
	k8s.io/klog v0.3.1
	k8s.io/kube-openapi v0.0.0-20190228160746-b3a7cee44a30
	k8s.io/utils v0.0.0-20190221042446-c2654d5206da
)

replace (
	golang.org/x/sync => golang.org/x/sync v0.0.0-20181108010431-42b317875d0f
	golang.org/x/sys => golang.org/x/sys v0.0.0-20190209173611-3b5209105503
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190313210603-aa82965741a9
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/apiserver => ../apiserver
	k8s.io/client-go => ../client-go
	k8s.io/code-generator => ../code-generator
	k8s.io/component-base => ../component-base
	k8s.io/kube-aggregator => ../kube-aggregator
)
