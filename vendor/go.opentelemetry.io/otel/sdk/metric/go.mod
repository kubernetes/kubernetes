module go.opentelemetry.io/otel/sdk/metric

go 1.14

replace go.opentelemetry.io/otel => ../..

replace go.opentelemetry.io/otel/bridge/opencensus => ../../bridge/opencensus

replace go.opentelemetry.io/otel/bridge/opentracing => ../../bridge/opentracing

replace go.opentelemetry.io/otel/example/jaeger => ../../example/jaeger

replace go.opentelemetry.io/otel/example/namedtracer => ../../example/namedtracer

replace go.opentelemetry.io/otel/example/opencensus => ../../example/opencensus

replace go.opentelemetry.io/otel/example/otel-collector => ../../example/otel-collector

replace go.opentelemetry.io/otel/example/prom-collector => ../../example/prom-collector

replace go.opentelemetry.io/otel/example/prometheus => ../../example/prometheus

replace go.opentelemetry.io/otel/example/zipkin => ../../example/zipkin

replace go.opentelemetry.io/otel/exporters/metric/prometheus => ../../exporters/metric/prometheus

replace go.opentelemetry.io/otel/exporters/otlp => ../../exporters/otlp

replace go.opentelemetry.io/otel/exporters/stdout => ../../exporters/stdout

replace go.opentelemetry.io/otel/exporters/trace/jaeger => ../../exporters/trace/jaeger

replace go.opentelemetry.io/otel/exporters/trace/zipkin => ../../exporters/trace/zipkin

replace go.opentelemetry.io/otel/internal/tools => ../../internal/tools

replace go.opentelemetry.io/otel/metric => ../../metric

replace go.opentelemetry.io/otel/oteltest => ../../oteltest

replace go.opentelemetry.io/otel/sdk => ../

replace go.opentelemetry.io/otel/sdk/export/metric => ../export/metric

replace go.opentelemetry.io/otel/sdk/metric => ./

replace go.opentelemetry.io/otel/trace => ../../trace

require (
	github.com/benbjohnson/clock v1.0.3 // do not upgrade to v1.1.x because it would require Go >= 1.15
	github.com/stretchr/testify v1.7.0
	go.opentelemetry.io/otel v0.20.0
	go.opentelemetry.io/otel/metric v0.20.0
	go.opentelemetry.io/otel/sdk v0.20.0
	go.opentelemetry.io/otel/sdk/export/metric v0.20.0
)
