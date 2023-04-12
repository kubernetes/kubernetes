// This is a generated file. Do not edit directly.

module k8s.io/kms

go 1.20

require (
	github.com/gogo/protobuf v1.3.2
	google.golang.org/grpc v1.51.0
	k8s.io/apimachinery v0.27.0-rc.1
	k8s.io/client-go v0.27.0-rc.1
	k8s.io/klog/v2 v2.90.1
)

require (
	github.com/go-logr/logr v1.2.3 // indirect
	github.com/golang/protobuf v1.5.3 // indirect
	golang.org/x/net v0.8.0 // indirect
	golang.org/x/sys v0.6.0 // indirect
	golang.org/x/text v0.8.0 // indirect
	golang.org/x/time v0.0.0-20220210224613-90d013bbcef8 // indirect
	google.golang.org/genproto v0.0.0-20220502173005-c8bf987b8c21 // indirect
	google.golang.org/protobuf v1.28.1 // indirect
	k8s.io/utils v0.0.0-20230313181309-38a27ef9d749 // indirect
)

replace (
	github.com/onsi/ginkgo/v2 => github.com/openshift/onsi-ginkgo/v2 v2.6.1-0.20230317131656-c62d9de5a460
	github.com/openshift/api => github.com/bertinatto/api v0.0.0-20230410165344-8f9c526f4e37
	github.com/openshift/apiserver-library-go => github.com/bertinatto/apiserver-library-go v0.0.0-20230410181057-513cffc083b1
	github.com/openshift/client-go => github.com/bertinatto/client-go v0.0.0-20230410172026-8ce025ee6689
	github.com/openshift/library-go => github.com/bertinatto/library-go v0.0.0-20230410173146-d901f4338cf7
	k8s.io/api => ../api
	k8s.io/apiextensions-apiserver => ../apiextensions-apiserver
	k8s.io/apimachinery => ../apimachinery
	k8s.io/apiserver => ../apiserver
	k8s.io/client-go => ../client-go
	k8s.io/code-generator => ../code-generator
	k8s.io/component-base => ../component-base
	k8s.io/component-helpers => ../component-helpers
	k8s.io/kms => ../kms
	k8s.io/kube-aggregator => ../kube-aggregator
)
