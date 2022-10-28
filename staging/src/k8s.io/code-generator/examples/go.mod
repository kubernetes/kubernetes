// This is a submodule to isolate k8s.io/code-generator from k8s.io/{api,apimachinery,client-go} dependencies in generated code

module k8s.io/code-generator/examples

go 1.16

require (
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/kube-openapi v0.0.0-20221012153701-172d655c2280
)

replace (
	k8s.io/api => ../../api
	k8s.io/apimachinery => ../../apimachinery
	k8s.io/client-go => ../../client-go
)

replace google.golang.org/grpc => github.com/liggitt/grpc-go v1.51.0-dev.0.20221027215202-2901f263bef4

replace google.golang.org/api => github.com/liggitt/google-api-go-client v0.101.1-0.20221028054038-b4aad6d92467

replace go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc => github.com/liggitt/opentelemetry-go-contrib/instrumentation/google.golang.org/grpc/otelgrpc v0.36.5-0.20221027221714-25c81eb49c35

replace github.com/google/cadvisor => github.com/liggitt/cadvisor v0.45.1-0.20221027232935-03ec2fc20e12
