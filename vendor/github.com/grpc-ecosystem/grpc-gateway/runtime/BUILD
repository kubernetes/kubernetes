load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "context.go",
        "convert.go",
        "doc.go",
        "errors.go",
        "fieldmask.go",
        "handler.go",
        "marshal_httpbodyproto.go",
        "marshal_json.go",
        "marshal_jsonpb.go",
        "marshal_proto.go",
        "marshaler.go",
        "marshaler_registry.go",
        "mux.go",
        "pattern.go",
        "proto2_convert.go",
        "proto_errors.go",
        "query.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/grpc-ecosystem/grpc-gateway/runtime",
    importpath = "github.com/grpc-ecosystem/grpc-gateway/runtime",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/golang/protobuf/jsonpb:go_default_library",
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/github.com/golang/protobuf/protoc-gen-go/generator:go_default_library",
        "//vendor/github.com/golang/protobuf/ptypes/any:go_default_library",
        "//vendor/github.com/golang/protobuf/ptypes/duration:go_default_library",
        "//vendor/github.com/golang/protobuf/ptypes/timestamp:go_default_library",
        "//vendor/github.com/golang/protobuf/ptypes/wrappers:go_default_library",
        "//vendor/github.com/grpc-ecosystem/grpc-gateway/internal:go_default_library",
        "//vendor/github.com/grpc-ecosystem/grpc-gateway/utilities:go_default_library",
        "//vendor/google.golang.org/genproto/googleapis/api/httpbody:go_default_library",
        "//vendor/google.golang.org/genproto/protobuf/field_mask:go_default_library",
        "//vendor/google.golang.org/grpc/codes:go_default_library",
        "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        "//vendor/google.golang.org/grpc/metadata:go_default_library",
        "//vendor/google.golang.org/grpc/status:go_default_library",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
