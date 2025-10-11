regenerate:
	go install k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo/internal/third_party/github.com/gogo/protobuf/protoc-gen-gogo
	go install k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo/internal/third_party/github.com/gogo/protobuf/protoc-min-version
	protoc-min-version --version="3.0.0" --proto_path=../../../../../../../../../../../../../:../../../protobuf/:. --gogo_out=. a.proto