module k8s.io/kms/plugins/mock

go 1.22.0

require (
	github.com/ThalesIgnite/crypto11 v1.2.5
	k8s.io/kms v0.0.0-00010101000000-000000000000
)

require (
	github.com/gogo/protobuf v1.3.2 // indirect
	github.com/miekg/pkcs11 v1.0.3-0.20190429190417-a667d056470f // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/thales-e-security/pool v0.0.2 // indirect
	golang.org/x/net v0.29.0 // indirect
	golang.org/x/sys v0.25.0 // indirect
	golang.org/x/text v0.18.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20240924160255-9d4c2d233b61 // indirect
	google.golang.org/grpc v1.67.0 // indirect
	google.golang.org/protobuf v1.34.2 // indirect
)

replace k8s.io/kms => ../../../../kms

replace github.com/google/cadvisor => github.com/openshift/google-cadvisor v0.49.0-openshift-4.17-2

replace github.com/onsi/ginkgo/v2 => github.com/openshift/onsi-ginkgo/v2 v2.6.1-0.20241008152707-25bf9f14db44
