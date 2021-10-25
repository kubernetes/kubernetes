// This is a submodule to isolate k8s.io/code-generator from k8s.io/{api,apimachinery,client-go} dependencies in generated code

module k8s.io/code-generator/examples

go 1.16

require (
	github.com/evanphx/json-patch v0.5.2 // indirect
	golang.org/x/oauth2 v0.0.0-20211005180243-6b3c2da341f1 // indirect
	golang.org/x/time v0.0.0-20210723032227-1f47c861a9ac // indirect
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/kube-openapi v0.0.0-20210817084001-7fbd8d59e5b8
)

replace (
	k8s.io/api => ../../../../../_vmod/k8s.io/api
	k8s.io/apimachinery => ../../../../../_vmod/k8s.io/apimachinery
	k8s.io/client-go => ../../../../../_vmod/k8s.io/client-go
)

replace sigs.k8s.io/json => github.com/liggitt/json v0.0.0-20211020163728-48258682683b
