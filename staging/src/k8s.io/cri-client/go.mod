// This is a generated file. Do not edit directly.

module k8s.io/cri-client

go 1.26.0

godebug default=go1.26

require (
	github.com/Microsoft/go-winio v0.6.2
	github.com/fsnotify/fsnotify v1.9.0
	github.com/stretchr/testify v1.12.1
	go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc v0.71.0
	go.opentelemetry.io/otel v1.46.0
	go.opentelemetry.io/otel/sdk v1.46.0
	go.opentelemetry.io/otel/trace v1.46.0
	golang.org/x/sys v0.47.0
	google.golang.org/grpc v1.83.2
	google.golang.org/protobuf v1.36.12
	k8s.io/component-base v0.0.0
	k8s.io/cri-api v0.0.0
	k8s.io/klog/v2 v2.140.0
	k8s.io/utils v0.0.0-20260626114624-be93311217bd
)

require (
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/go-logr/logr v1.4.4 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/google/uuid v1.6.0 // indirect
	go.opentelemetry.io/auto/sdk v1.2.1 // indirect
	go.opentelemetry.io/otel/metric v1.46.0 // indirect
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/net v0.58.0 // indirect
	golang.org/x/text v0.41.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260825221802-da73d73af1c5 // indirect
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/component-base => ../component-base
	k8s.io/cri-api => ../cri-api
)
