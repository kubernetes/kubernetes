module k8s.io/kubernetes/hack/tools/instrumentation

go 1.26.0

require (
	github.com/google/go-cmp v0.7.0
	github.com/prometheus/client_golang v1.23.2
	github.com/spf13/pflag v1.0.10
	go.yaml.in/yaml/v2 v2.4.3
	k8s.io/component-base v0.0.0-00010101000000-000000000000
)

require (
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/blang/semver/v4 v4.0.0 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/kylelemons/godebug v1.1.0 // indirect
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822 // indirect
	github.com/prometheus/client_model v0.6.2 // indirect
	github.com/prometheus/common v0.67.5 // indirect
	github.com/prometheus/procfs v0.19.2 // indirect
	go.opentelemetry.io/otel v1.43.0 // indirect
	go.opentelemetry.io/otel/trace v1.43.0 // indirect
	golang.org/x/sys v0.43.0 // indirect
	google.golang.org/protobuf v1.36.12-0.20260120151049-f2248ac996af // indirect
	k8s.io/apimachinery v0.0.0 // indirect
	k8s.io/klog/v2 v2.140.0 // indirect
)

replace k8s.io/apimachinery => ../../../staging/src/k8s.io/apimachinery

replace k8s.io/api => ../../../staging/src/k8s.io/api

replace k8s.io/component-base => ../../../staging/src/k8s.io/component-base

replace k8s.io/kubernetes => ../../../
