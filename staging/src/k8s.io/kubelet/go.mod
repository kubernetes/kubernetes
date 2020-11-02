// This is a generated file. Do not edit directly.

module k8s.io/kubelet

go 1.15

require (
	github.com/gogo/protobuf v1.3.1
	golang.org/x/net v0.0.0-20201022231255-08b38378de70
	google.golang.org/grpc v1.27.1
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/component-base v0.0.0
)

replace (
	golang.org/x/sys => golang.org/x/sys v0.0.0-20200930185726-fdedc70b468f
	google.golang.org/grpc => google.golang.org/grpc v1.27.0
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/component-base => ../component-base
	k8s.io/kubelet => ../kubelet
)
