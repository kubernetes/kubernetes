// This is a generated file. Do not edit directly.

module k8s.io/sample-apiserver

go 1.12

require (
	github.com/google/gofuzz v0.0.0-20170612174753-24818f796faf
	github.com/inconshreveable/mousetrap v1.0.0 // indirect
	github.com/spf13/cobra v0.0.0-20180319062004-c439c4fa0937
	k8s.io/apimachinery v0.0.0
	k8s.io/apiserver v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/code-generator v0.0.0
	k8s.io/component-base v0.0.0
	k8s.io/klog v0.3.1
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
	k8s.io/sample-apiserver => ../sample-apiserver
)
