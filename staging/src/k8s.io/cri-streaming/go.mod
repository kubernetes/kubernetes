// This is a generated file. Do not edit directly.

module k8s.io/cri-streaming

go 1.26.0

godebug default=go1.26

require (
	github.com/emicklei/go-restful/v3 v3.13.0
	github.com/gorilla/websocket v1.5.4-0.20250319132907-e064f32e3674
	github.com/stretchr/testify v1.11.1
	go.uber.org/goleak v1.3.0
	google.golang.org/grpc v1.79.3
	k8s.io/cri-api v0.0.0
	k8s.io/klog/v2 v2.140.0
	k8s.io/streaming v0.0.0
	k8s.io/utils v0.0.0-20260210185600-b8788abfbbc2
)

require (
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/moby/spdystream v0.5.0 // indirect
	github.com/pmezard/go-difflib v1.0.1-0.20181226105442-5d4384ee4fb2 // indirect
	go.opentelemetry.io/otel/metric v1.40.0 // indirect
	go.opentelemetry.io/otel/sdk v1.40.0 // indirect
	go.opentelemetry.io/otel/trace v1.40.0 // indirect
	golang.org/x/net v0.49.0 // indirect
	golang.org/x/sys v0.40.0 // indirect
	golang.org/x/text v0.33.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260128011058-8636f8732409 // indirect
	google.golang.org/protobuf v1.36.12-0.20260120151049-f2248ac996af // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace (
	k8s.io/cri-api => ../cri-api
	k8s.io/streaming => ../streaming
)
