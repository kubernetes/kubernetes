// This is a generated file. Do not edit directly.

module k8s.io/kube-controller-manager

go 1.13

require (
	k8s.io/apimachinery v0.0.0
	k8s.io/component-base v0.0.0
)

replace (
	github.com/sirupsen/logrus => github.com/sirupsen/logrus v1.4.2
	golang.org/x/sys => golang.org/x/sys v0.0.0-20190813064441-fde4db37ae7a // pinned to release-branch.go1.13
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190821162956-65e3620a7ae7 // pinned to release-branch.go1.13
	google.golang.org/protobuf => google.golang.org/protobuf v1.23.0
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/component-base => ../component-base
	k8s.io/kube-controller-manager => ../kube-controller-manager
)
