// This is a generated file. Do not edit directly.

module k8s.io/component-base

go 1.16

require (
	github.com/blang/semver v3.5.1+incompatible
	github.com/go-logr/logr v1.2.0
	github.com/go-logr/zapr v1.2.0
	github.com/google/go-cmp v0.5.5
	github.com/moby/term v0.0.0-20210619224110-3f7ff695adc6
	github.com/prometheus/client_golang v1.12.0
	github.com/prometheus/client_model v0.2.0
	github.com/prometheus/common v0.32.1
	github.com/prometheus/procfs v0.7.3
	github.com/spf13/cobra v1.2.1
	github.com/spf13/pflag v1.0.5
	github.com/stretchr/testify v1.7.0
	go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp v0.20.0
	go.opentelemetry.io/otel v0.20.0
	go.opentelemetry.io/otel/exporters/otlp v0.20.0
	go.opentelemetry.io/otel/sdk v0.20.0
	go.opentelemetry.io/otel/trace v0.20.0
	go.uber.org/zap v1.19.0
	golang.org/x/sys v0.0.0-20220114195835-da31bd327af9
	k8s.io/apimachinery v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/klog/v2 v2.40.1
	k8s.io/utils v0.0.0-20211208161948-7d6a63dca704
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/apiserver => ../apiserver
	k8s.io/client-go => ../client-go
	k8s.io/component-base => ../component-base
)
