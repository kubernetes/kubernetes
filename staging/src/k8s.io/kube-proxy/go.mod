// This is a generated file. Do not edit directly.

module k8s.io/kube-proxy

go 1.17

require (
	github.com/gogo/protobuf v1.3.2 // indirect
	github.com/google/go-cmp v0.5.5 // indirect
	github.com/google/gofuzz v1.1.0 // indirect
	github.com/json-iterator/go v1.1.11 // indirect
	golang.org/x/net v0.0.0-20210813160813-60bc85c4be6d // indirect
	gopkg.in/yaml.v2 v2.4.0 // indirect
	k8s.io/klog/v2 v2.9.0 // indirect
	k8s.io/utils v0.0.0-20210819203725-bdf08cb9a70a // indirect
	sigs.k8s.io/structured-merge-diff/v4 v4.1.2 // indirect
	sigs.k8s.io/yaml v1.2.0 // indirect
)

require (
	github.com/go-logr/logr v0.4.0 // indirect
	github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd // indirect
	github.com/modern-go/reflect2 v1.0.1 // indirect
	golang.org/x/text v0.3.6 // indirect
	gopkg.in/inf.v0 v0.9.1 // indirect
	k8s.io/apimachinery v0.0.0
	k8s.io/component-base v0.0.0
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/component-base => ../component-base
	k8s.io/kube-proxy => ../kube-proxy
)
