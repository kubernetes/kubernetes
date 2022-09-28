// This is a submodule to isolate k8s.io/code-generator from k8s.io/{api,apimachinery,client-go} dependencies in generated code

module k8s.io/code-generator/examples

go 1.16

require (
	k8s.io/api v0.25.0
	k8s.io/apimachinery v0.25.0
	k8s.io/client-go v0.0.0
	k8s.io/kube-openapi v0.0.0-20220803162953-67bda5d908f1
)

replace (
	github.com/onsi/ginkgo/v2 => github.com/soltysh/ginkgo/v2 v2.1.5-0.20220819125456-719bfd56933e
	k8s.io/api => ../../api
	k8s.io/apimachinery => ../../apimachinery
	k8s.io/client-go => ../../client-go
)
