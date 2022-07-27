module go.etcd.io/etcd/client/v2

go 1.16

require (
	github.com/json-iterator/go v1.1.11
	github.com/modern-go/reflect2 v1.0.1
	go.etcd.io/etcd/api/v3 v3.5.4
	go.etcd.io/etcd/client/pkg/v3 v3.5.4
)

replace (
	go.etcd.io/etcd/api/v3 => ../../api
	go.etcd.io/etcd/client/pkg/v3 => ../pkg
)

// Bad imports are sometimes causing attempts to pull that code.
// This makes the error more explicit.
replace (
	go.etcd.io/etcd => ./FORBIDDEN_DEPENDENCY
	go.etcd.io/etcd/pkg/v3 => ./FORBIDDED_DEPENDENCY
	go.etcd.io/etcd/tests/v3 => ./FORBIDDEN_DEPENDENCY
	go.etcd.io/etcd/v3 => ./FORBIDDEN_DEPENDENCY
)
