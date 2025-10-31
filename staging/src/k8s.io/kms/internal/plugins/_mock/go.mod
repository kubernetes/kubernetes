module k8s.io/kms/plugins/mock

go 1.25.0

godebug default=go1.25

require (
	github.com/ThalesIgnite/crypto11 v1.2.5
	k8s.io/kms v0.0.0-00010101000000-000000000000
)

require (
	github.com/miekg/pkcs11 v1.0.3-0.20190429190417-a667d056470f // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/thales-e-security/pool v0.0.2 // indirect
	golang.org/x/net v0.43.0 // indirect
	golang.org/x/sys v0.37.0 // indirect
	golang.org/x/text v0.29.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20250818200422-3122310a409c // indirect
	google.golang.org/grpc v1.75.0-dev // indirect
	google.golang.org/protobuf v1.36.8 // indirect
)

replace k8s.io/kms => ../../../../kms

replace k8s.io/kube-openapi => github.com/pohly/kube-openapi v0.0.0-20251031162619-7c7e24fa61bc
