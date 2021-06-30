module go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp

go 1.14

replace go.opentelemetry.io/contrib => ../../../..

require (
	github.com/felixge/httpsnoop v1.0.1
	github.com/stretchr/testify v1.7.0
	go.opentelemetry.io/contrib v0.20.0
	go.opentelemetry.io/otel v0.20.0
	go.opentelemetry.io/otel/metric v0.20.0
	go.opentelemetry.io/otel/oteltest v0.20.0
	go.opentelemetry.io/otel/trace v0.20.0
)
