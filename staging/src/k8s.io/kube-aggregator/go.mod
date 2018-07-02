module k8s.io/kube-aggregator

require (
	k8s.io/api v1.12.0
	k8s.io/apimachinery v1.12.0
	k8s.io/apiserver v1.12.0
	k8s.io/client-go v1.12.0
)

replace (
	k8s.io/api v1.12.0 => ../api
	k8s.io/apimachinery v1.12.0 => ../apimachinery
	k8s.io/apiserver v1.12.0 => ../apiserver
	k8s.io/client-go v1.12.0 => ../client-go
)

require (
	github.com/emicklei/go-restful v0.0.0-20170410110728-ff4f55a20633
	github.com/go-openapi/spec v0.0.0-20180213232550-1de3e0542de6
	github.com/gogo/protobuf v0.0.0-20170330071051-c0656edd0d9e
	github.com/golang/glog v0.0.0-20141105023935-44145f04b68c
	github.com/inconshreveable/mousetrap v1.0.0 // indirect
	github.com/spf13/cobra v0.0.0-20180319062004-c439c4fa0937
	github.com/spf13/pflag v1.0.1
	github.com/stretchr/testify v0.0.0-20180319223459-c679ae2cc0cb
	golang.org/x/net v0.0.0-20170809000501-1c05540f6879
	k8s.io/kube-openapi v0.0.0-20180620173706-91cfa479c814
)
