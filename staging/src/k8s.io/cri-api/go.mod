// This is a generated file. Do not edit directly.

module k8s.io/cri-api

go 1.19

require (
	github.com/gogo/protobuf v1.3.2
	github.com/stretchr/testify v1.8.0
	google.golang.org/grpc v1.51.0-dev.0.20221027215202-2901f263bef4
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/golang/protobuf v1.5.2 // indirect
	github.com/google/go-cmp v0.5.9 // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/niemeyer/pretty v0.0.0-20200227124842-a10e7caefd8e // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	golang.org/x/net v0.1.1-0.20221027164007-c63010009c80 // indirect
	golang.org/x/sys v0.1.0 // indirect
	golang.org/x/text v0.4.0 // indirect
	google.golang.org/genproto v0.0.0-20220502173005-c8bf987b8c21 // indirect
	google.golang.org/protobuf v1.28.1 // indirect
	gopkg.in/check.v1 v1.0.0-20200227125254-8fa46927fb4f // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace k8s.io/cri-api => ../cri-api

replace google.golang.org/grpc => github.com/liggitt/grpc-go v1.51.0-dev.0.20221027215202-2901f263bef4

replace google.golang.org/api => github.com/liggitt/google-api-go-client v0.101.1-0.20221028054038-b4aad6d92467

replace go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc => github.com/liggitt/opentelemetry-go-contrib/instrumentation/google.golang.org/grpc/otelgrpc v0.36.5-0.20221027221714-25c81eb49c35

replace github.com/google/cadvisor => github.com/liggitt/cadvisor v0.45.1-0.20221027232935-03ec2fc20e12
