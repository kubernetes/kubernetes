regenerate:
	go install k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo/internal/third_party/github.com/gogo/protobuf/protoc-gen-gogo
	go install k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo/internal/third_party/github.com/gogo/protobuf/protoc-min-version
	protoc-min-version --version="3.0.0" --gogo_out=\
	Mgoogle/protobuf/any.proto=k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo/internal/third_party/github.com/gogo/protobuf/types,\
	Mgoogle/protobuf/api.proto=k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo/internal/third_party/github.com/gogo/protobuf/types,\
	Mgoogle/protobuf/duration.proto=k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo/internal/third_party/github.com/gogo/protobuf/types,\
	Mgoogle/protobuf/empty.proto=k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo/internal/third_party/github.com/gogo/protobuf/types,\
	Mgoogle/protobuf/field_mask.proto=k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo/internal/third_party/github.com/gogo/protobuf/types,\
	Mgoogle/protobuf/source_context.proto=k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo/internal/third_party/github.com/gogo/protobuf/types,\
	Mgoogle/protobuf/struct.proto=k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo/internal/third_party/github.com/gogo/protobuf/types,\
	Mgoogle/protobuf/timestamp.proto=k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo/internal/third_party/github.com/gogo/protobuf/types,\
	Mgoogle/protobuf/type.proto=k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo/internal/third_party/github.com/gogo/protobuf/types,\
	Mgoogle/protobuf/wrappers.proto=k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo/internal/third_party/github.com/gogo/protobuf/types,\
	:. --proto_path=../../../../../../../../../../../../:../../protobuf/:. issue438.proto
