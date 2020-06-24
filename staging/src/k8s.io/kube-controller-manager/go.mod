// This is a generated file. Do not edit directly.

module k8s.io/kube-controller-manager

go 1.14

require (
	k8s.io/apimachinery v0.0.0
	k8s.io/component-base v0.0.0
)

replace (
	golang.org/x/sys => golang.org/x/sys v0.0.0-20200201011859-915c9c3d4ccf // pinned to release-branch.go1.14-std
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/component-base => ../component-base
	k8s.io/kube-controller-manager => ../kube-controller-manager
)
