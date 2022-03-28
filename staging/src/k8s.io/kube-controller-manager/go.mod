// This is a generated file. Do not edit directly.

module k8s.io/kube-controller-manager

go 1.16

require (
	k8s.io/apimachinery v0.0.0
	k8s.io/cloud-provider v0.0.0
	k8s.io/controller-manager v0.0.0
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/apiserver => ../apiserver
	k8s.io/client-go => ../client-go
	k8s.io/cloud-provider => ../cloud-provider
	k8s.io/component-base => ../component-base
	k8s.io/component-helpers => ../component-helpers
	k8s.io/controller-manager => ../controller-manager
	k8s.io/kube-controller-manager => ../kube-controller-manager
	k8s.io/kube-openapi => github.com/jefftree/kube-openapi v0.0.8-gnostic.0.20220328150837-013580d8b582
)
