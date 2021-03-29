// This is a generated file. Do not edit directly.

module k8s.io/code-generator

go 1.13

require (
	github.com/emicklei/go-restful v2.9.5+incompatible // indirect
	github.com/go-openapi/jsonreference v0.19.3 // indirect
	github.com/go-openapi/spec v0.19.3 // indirect
	github.com/gogo/protobuf v1.3.2
	github.com/google/go-cmp v0.3.0 // indirect
	github.com/json-iterator/go v1.1.8 // indirect
	github.com/mailru/easyjson v0.7.0 // indirect
	github.com/spf13/pflag v1.0.5
	github.com/stretchr/testify v1.4.0 // indirect
	golang.org/x/net v0.0.0-20201110031124-69a78807bb2b // indirect
	k8s.io/gengo v0.0.0-20200114144118-36b2048a9120
	k8s.io/klog v1.0.0
	k8s.io/kube-openapi v0.0.0-20200410145947-61e04a5be9a6 // release-1.18
	sigs.k8s.io/yaml v1.2.0 // indirect
)

replace (
	golang.org/x/crypto => golang.org/x/crypto v0.0.0-20200220183623-bac4c82f6975
	golang.org/x/text => golang.org/x/text v0.3.2
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190821162956-65e3620a7ae7 // pinned to release-branch.go1.13
	k8s.io/code-generator => ../code-generator
)
