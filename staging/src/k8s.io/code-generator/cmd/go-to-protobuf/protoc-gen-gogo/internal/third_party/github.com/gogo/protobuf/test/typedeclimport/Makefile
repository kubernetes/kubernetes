regenerate:
	go install k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo/internal/third_party/github.com/gogo/protobuf/protoc-gen-gogo
	protoc-min-version --version="3.0.0" --gogo_out=. --proto_path=../../../../../../../../../../../../:../../protobuf/:. typedeclimport.proto
	protoc-min-version --version="3.0.0" --gogo_out=. --proto_path=../../../../../../../../../../../../:../../protobuf/:. ./subpkg/subpkg.proto