regenerate:
	protoc --proto_path=.:$(GOPATH)/src/k8s.io/kubernetes/staging/src/:$(GOPATH)/src/k8s.io/kubernetes/staging/src/k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo/internal/third_party/github.com/gogo/protobuf/protobuf/ \
	--gogo_out=. issue312.proto 
	(cd events && make regenerate)
