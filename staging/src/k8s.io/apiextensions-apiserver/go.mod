module k8s.io/apiextensions-apiserver

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
	github.com/asaskevich/govalidator v0.0.0-20160715170612-593d64559f76 // indirect
	github.com/coreos/etcd v0.0.0-20180102212956-95a726a27e09
	github.com/ghodss/yaml v0.0.0-20150909031657-73d445a93680
	github.com/go-openapi/analysis v0.0.0-20160815203709-b44dc874b601 // indirect
	github.com/go-openapi/errors v0.0.0-20160704190347-d24ebc2075ba // indirect
	github.com/go-openapi/loads v0.0.0-20170520182102-a80dea3052f0 // indirect
	github.com/go-openapi/runtime v0.0.0-20160704190703-11e322eeecc1 // indirect
	github.com/go-openapi/spec v0.0.0-20180213232550-1de3e0542de6
	github.com/go-openapi/strfmt v0.0.0-20160812050534-d65c7fdb29ec
	github.com/go-openapi/validate v0.0.0-20171117174350-d509235108fc
	github.com/gogo/protobuf v0.0.0-20170330071051-c0656edd0d9e
	github.com/golang/glog v0.0.0-20141105023935-44145f04b68c
	github.com/google/gofuzz v0.0.0-20161122191042-44d81051d367
	github.com/inconshreveable/mousetrap v1.0.0 // indirect
	github.com/pborman/uuid v0.0.0-20150603214016-ca53cad383ca
	github.com/spf13/cobra v0.0.0-20180319062004-c439c4fa0937
	github.com/stretchr/testify v0.0.0-20180319223459-c679ae2cc0cb
)
