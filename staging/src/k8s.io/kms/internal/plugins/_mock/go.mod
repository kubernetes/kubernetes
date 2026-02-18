module k8s.io/kms/plugins/mock

go 1.26.0

godebug default=go1.26

require (
	github.com/ThalesIgnite/crypto11 v1.2.5
	k8s.io/kms v0.0.0-00010101000000-000000000000
)

require (
	github.com/miekg/pkcs11 v1.0.3-0.20190429190417-a667d056470f // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/thales-e-security/pool v0.0.2 // indirect
	golang.org/x/net v0.50.0 // indirect
	golang.org/x/sys v0.41.0 // indirect
	golang.org/x/text v0.34.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260128011058-8636f8732409 // indirect
	google.golang.org/grpc v1.78.0 // indirect
	google.golang.org/protobuf v1.36.11 // indirect
)

replace k8s.io/kms => ../../../../kms

replace go.etcd.io/etcd/api/v3 => github.com/liggitt/etcd/api/v3 v3.6.0-alpha.0.0.20260217204530-004f720483b2

replace go.etcd.io/etcd/client/pkg/v3 => github.com/liggitt/etcd/client/pkg/v3 v3.6.0-alpha.0.0.20260217204530-004f720483b2

replace go.etcd.io/etcd/client/v3 => github.com/liggitt/etcd/client/v3 v3.6.0-alpha.0.0.20260217204530-004f720483b2

replace go.etcd.io/etcd/pkg/v3 => github.com/liggitt/etcd/pkg/v3 v3.6.0-alpha.0.0.20260217204530-004f720483b2

replace go.etcd.io/etcd/server/v3 => github.com/liggitt/etcd/server/v3 v3.6.0-alpha.0.0.20260217204530-004f720483b2
