module k8s.io/sample-apiserver

require (
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
	github.com/golang/glog v0.0.0-20141105023935-44145f04b68c
	github.com/google/gofuzz v0.0.0-20161122191042-44d81051d367
	github.com/inconshreveable/mousetrap v1.0.0 // indirect
	github.com/spf13/cobra v0.0.0-20180319062004-c439c4fa0937
)
