// This is a submodule to isolate k8s.io/code-generator from k8s.io/{api,apimachinery,client-go} dependencies in generated code

module k8s.io/code-generator/examples

go 1.16

require (
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/kube-openapi v0.0.0-20221012153701-172d655c2280
)

replace (
	k8s.io/api => ../../api
	k8s.io/apimachinery => ../../apimachinery
	k8s.io/client-go => ../../client-go
)

replace k8s.io/klog/v2 => github.com/pohly/klog/v2 v2.40.2-0.20221123181939-72bb78b1e07a

replace github.com/go-logr/logr => github.com/pohly/logr v1.0.1-0.20221123084031-5a74e449aa62
