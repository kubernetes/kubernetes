// This is a generated file. Do not edit directly.

module k8s.io/kube-aggregator

go 1.12

require (
	github.com/davecgh/go-spew v1.1.1
	github.com/emicklei/go-restful v2.9.5+incompatible
	github.com/ghodss/yaml v0.0.0-20180820084758-c7ce16629ff4 // indirect
	github.com/go-openapi/spec v0.19.2
	github.com/gogo/protobuf v1.2.2-0.20190723190241-65acae22fc9d
	github.com/prometheus/client_golang v0.9.2
	github.com/spf13/cobra v0.0.5
	github.com/spf13/pflag v1.0.3
	github.com/stretchr/testify v1.3.0
	golang.org/x/net v0.0.0-20190812203447-cdfb69ac37fc
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/apiserver v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/code-generator v0.0.0
	k8s.io/component-base v0.0.0
	k8s.io/klog v0.4.0
	k8s.io/kube-openapi v0.0.0-20190709113604-33be087ad058
	k8s.io/utils v0.0.0-20190801114015-581e00157fb1
)

replace (
	golang.org/x/crypto => golang.org/x/crypto v0.0.0-20181025213731-e84da0312774
	golang.org/x/lint => golang.org/x/lint v0.0.0-20181217174547-8f45f776aaf1
	golang.org/x/sync => golang.org/x/sync v0.0.0-20181108010431-42b317875d0f
	golang.org/x/sys => golang.org/x/sys v0.0.0-20190209173611-3b5209105503
	golang.org/x/text => golang.org/x/text v0.3.1-0.20181227161524-e6919f6577db
	google.golang.org/genproto => google.golang.org/genproto v0.0.0-20170731182057-09f6ed296fc6
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/apiserver => ../apiserver
	k8s.io/client-go => ../client-go
	k8s.io/code-generator => ../code-generator
	k8s.io/component-base => ../component-base
	k8s.io/kube-aggregator => ../kube-aggregator
)
