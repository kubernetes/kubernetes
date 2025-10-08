regenerate:
	go install k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo/internal/third_party/github.com/gogo/protobuf/protoc-gen-gogo
	protoc --gogo_out=. --proto_path=../../../../../../../../../../../../:../../protobuf/:. *.proto
