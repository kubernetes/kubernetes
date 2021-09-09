// This is a generated file. Do not edit directly.

module k8s.io/kubelet

go 1.17

require (
	github.com/gogo/protobuf v1.3.2
	golang.org/x/net v0.0.0-20210813160813-60bc85c4be6d
	google.golang.org/grpc v1.38.0
)

require (
	github.com/go-logr/logr v0.4.0 // indirect
	github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd // indirect
	github.com/modern-go/reflect2 v1.0.1 // indirect
	golang.org/x/text v0.3.6 // indirect
	gopkg.in/inf.v0 v0.9.1 // indirect
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/component-base v0.0.0
)

require (
	github.com/golang/protobuf v1.5.2 // indirect
	github.com/google/go-cmp v0.5.5 // indirect
	github.com/google/gofuzz v1.1.0 // indirect
	github.com/json-iterator/go v1.1.11 // indirect
	golang.org/x/sys v0.0.0-20210820121016-41cdb8703e55 // indirect
	google.golang.org/genproto v0.0.0-20210602131652-f16073e35f0c // indirect
	google.golang.org/protobuf v1.26.0 // indirect
	gopkg.in/yaml.v2 v2.4.0 // indirect
	k8s.io/klog/v2 v2.9.0 // indirect
	k8s.io/utils v0.0.0-20210819203725-bdf08cb9a70a // indirect
	sigs.k8s.io/structured-merge-diff/v4 v4.1.2 // indirect
	sigs.k8s.io/yaml v1.2.0 // indirect
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/component-base => ../component-base
	k8s.io/kubelet => ../kubelet
)
