// This is a generated file. Do not edit directly.

module k8s.io/kubelet

go 1.13

require (
	github.com/gogo/protobuf v1.3.1
	golang.org/x/net v0.0.0-20191004110552-13f9640d40b9
	google.golang.org/grpc v1.26.0
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
)

replace (
	golang.org/x/sys => golang.org/x/sys v0.0.0-20190813064441-fde4db37ae7a // pinned to release-branch.go1.13
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190821162956-65e3620a7ae7 // pinned to release-branch.go1.13
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/kubelet => ../kubelet
)
