// This is a generated file. Do not edit directly.

module k8s.io/cri-streaming

go 1.26.0

godebug default=go1.26

require (
	github.com/emicklei/go-restful/v3 v3.13.0
	github.com/gorilla/websocket v1.5.4-0.20250319132907-e064f32e3674
	github.com/stretchr/testify v1.11.1
	go.uber.org/goleak v1.3.0
	google.golang.org/grpc v1.80.0
	k8s.io/cri-api v0.0.0
	k8s.io/klog/v2 v2.140.0
	k8s.io/streaming v0.0.0
	k8s.io/utils v0.0.0-20260210185600-b8788abfbbc2
)

require (
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/moby/spdystream v0.5.1 // indirect
	github.com/pmezard/go-difflib v1.0.1-0.20181226105442-5d4384ee4fb2 // indirect
	go.opentelemetry.io/otel/metric v1.43.0 // indirect
	go.opentelemetry.io/otel/sdk v1.43.0 // indirect
	go.opentelemetry.io/otel/trace v1.43.0 // indirect
	golang.org/x/net v0.53.0 // indirect
	golang.org/x/sys v0.43.0 // indirect
	golang.org/x/text v0.36.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260406210006-6f92a3bedf2d // indirect
	google.golang.org/protobuf v1.36.12-0.20260120151049-f2248ac996af // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace (
	k8s.io/api => ../api
	k8s.io/apiextensions-apiserver => ../apiextensions-apiserver
	k8s.io/apimachinery => ../apimachinery
	k8s.io/apiserver => ../apiserver
	k8s.io/cli-runtime => ../cli-runtime
	k8s.io/client-go => ../client-go
	k8s.io/cloud-provider => ../cloud-provider
	k8s.io/cluster-bootstrap => ../cluster-bootstrap
	k8s.io/code-generator => ../code-generator
	k8s.io/component-base => ../component-base
	k8s.io/component-helpers => ../component-helpers
	k8s.io/controller-manager => ../controller-manager
	k8s.io/cri-api => ../cri-api
	k8s.io/cri-client => ../cri-client
	k8s.io/cri-streaming => ../cri-streaming
	k8s.io/csi-translation-lib => ../csi-translation-lib
	k8s.io/dynamic-resource-allocation => ../dynamic-resource-allocation
	k8s.io/endpointslice => ../endpointslice
	k8s.io/externaljwt => ../externaljwt
	k8s.io/kms => ../kms
	k8s.io/kube-aggregator => ../kube-aggregator
	k8s.io/kube-controller-manager => ../kube-controller-manager
	k8s.io/kube-proxy => ../kube-proxy
	k8s.io/kube-scheduler => ../kube-scheduler
	k8s.io/kubectl => ../kubectl
	k8s.io/kubelet => ../kubelet
	k8s.io/metrics => ../metrics
	k8s.io/mount-utils => ../mount-utils
	k8s.io/pod-security-admission => ../pod-security-admission
	k8s.io/sample-apiserver => ../sample-apiserver
	k8s.io/sample-cli-plugin => ../sample-cli-plugin
	k8s.io/sample-controller => ../sample-controller
	k8s.io/streaming => ../streaming
)
