// This is a generated file. Do not edit directly.

module k8s.io/cluster-bootstrap

go 1.27.0

godebug default=go1.27

require (
	github.com/stretchr/testify v1.12.1
	gopkg.in/go-jose/go-jose.v2 v2.6.3
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/klog/v2 v2.140.0
)

require (
	github.com/fxamacker/cbor/v2 v2.9.1 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/x448/float16 v0.8.4 // indirect
	go.yaml.in/yaml/v2 v2.4.4 // indirect
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/crypto v0.54.0 // indirect
	golang.org/x/net v0.57.0 // indirect
	golang.org/x/text v0.40.0 // indirect
	gopkg.in/inf.v0 v0.9.1 // indirect
	k8s.io/kube-openapi v0.0.0-20260904170622-9ab3195f2a72 // indirect
	k8s.io/utils v0.0.0-20260626114624-be93311217bd // indirect
	sigs.k8s.io/json v0.0.0-20250730193827-2d320260d730 // indirect
	sigs.k8s.io/randfill v1.0.0 // indirect
	sigs.k8s.io/structured-merge-diff/v7 v7.0.0 // indirect
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/streaming => ../streaming
)
