// This is a generated file. Do not edit directly.

module k8s.io/cli-runtime

go 1.12

require (
	github.com/davecgh/go-spew v1.1.1
	github.com/emicklei/go-restful v0.0.0-20170410110728-ff4f55a20633 // indirect
	github.com/evanphx/json-patch v0.0.0-20190203023257-5858425f7550
	github.com/ghodss/yaml v0.0.0-20180820084758-c7ce16629ff4 // indirect
	github.com/go-openapi/jsonpointer v0.19.0 // indirect
	github.com/go-openapi/jsonreference v0.19.0 // indirect
	github.com/go-openapi/spec v0.17.2 // indirect
	github.com/go-openapi/swag v0.17.2 // indirect
	github.com/inconshreveable/mousetrap v1.0.0 // indirect
	github.com/pkg/errors v0.8.0
	github.com/spf13/cobra v0.0.0-20180319062004-c439c4fa0937
	github.com/spf13/pflag v1.0.1
	github.com/stretchr/testify v1.2.2
	golang.org/x/text v0.3.1-0.20181227161524-e6919f6577db
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/client-go v0.0.0
	sigs.k8s.io/kustomize v2.0.3+incompatible
	sigs.k8s.io/yaml v1.1.0
)

replace (
	golang.org/x/sync => golang.org/x/sync v0.0.0-20181108010431-42b317875d0f
	golang.org/x/sys => golang.org/x/sys v0.0.0-20190209173611-3b5209105503
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190313210603-aa82965741a9
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/cli-runtime => ../cli-runtime
	k8s.io/client-go => ../client-go
)
