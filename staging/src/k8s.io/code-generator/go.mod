// This is a generated file. Do not edit directly.

module k8s.io/code-generator

go 1.13

require (
	github.com/emicklei/go-restful v2.9.5+incompatible // indirect
	github.com/go-openapi/jsonreference v0.19.3 // indirect
	github.com/go-openapi/spec v0.19.3 // indirect
	github.com/gogo/protobuf v1.3.1
	github.com/google/go-cmp v0.3.0 // indirect
	github.com/json-iterator/go v1.1.8 // indirect
	github.com/mailru/easyjson v0.7.0 // indirect
	github.com/spf13/pflag v1.0.5
	github.com/stretchr/testify v1.4.0 // indirect
	golang.org/x/net v0.0.0-20191004110552-13f9640d40b9 // indirect
	golang.org/x/tools v0.0.0-20190920225731-5eefd052ad72 // indirect
	k8s.io/gengo v0.0.0-20200114144118-36b2048a9120
	k8s.io/klog v1.0.0
	k8s.io/kube-openapi v0.0.0-20200121204235-bf4fb3bd569c
	sigs.k8s.io/yaml v1.2.0 // indirect
)

replace (
	golang.org/x/sys => golang.org/x/sys v0.0.0-20190813064441-fde4db37ae7a // pinned to release-branch.go1.13
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190821162956-65e3620a7ae7 // pinned to release-branch.go1.13
	k8s.io/code-generator => ../code-generator
)
