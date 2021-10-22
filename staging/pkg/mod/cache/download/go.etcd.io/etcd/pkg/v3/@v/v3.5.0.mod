module go.etcd.io/etcd/pkg/v3

go 1.16

require (
	github.com/creack/pty v1.1.11
	github.com/dustin/go-humanize v1.0.0
	github.com/golang/protobuf v1.5.1 // indirect
	github.com/spf13/cobra v1.1.3
	github.com/spf13/pflag v1.0.5
	github.com/stretchr/testify v1.7.0
	go.etcd.io/etcd/client/pkg/v3 v3.5.0
	go.uber.org/zap v1.17.0
	google.golang.org/grpc v1.38.0
)

replace (
	go.etcd.io/etcd => ./FORBIDDEN_DEPENDENCY
	go.etcd.io/etcd/api/v3 => ./FORBIDDEN_DEPENDENCY
	go.etcd.io/etcd/client/pkg/v3 => ../client/pkg
	go.etcd.io/etcd/tests/v3 => ./FORBIDDEN_DEPENDENCY
	go.etcd.io/etcd/v3 => ./FORBIDDEN_DEPENDENCY
)
