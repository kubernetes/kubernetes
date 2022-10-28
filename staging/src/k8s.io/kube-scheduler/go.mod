// This is a generated file. Do not edit directly.

module k8s.io/kube-scheduler

go 1.16

require (
	github.com/google/go-cmp v0.5.5
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/component-base v0.0.0
	sigs.k8s.io/yaml v1.2.0
)

replace (
	golang.org/x/sys => golang.org/x/sys v0.0.0-20220908164124-27713097b956
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/component-base => ../component-base
	k8s.io/kube-scheduler => ../kube-scheduler
)
