load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["command.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/gogo/protobuf/vanity/command",
    importpath = "github.com/gogo/protobuf/vanity/command",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/gogo/protobuf/plugin/compare:go_default_library",
        "//vendor/github.com/gogo/protobuf/plugin/defaultcheck:go_default_library",
        "//vendor/github.com/gogo/protobuf/plugin/description:go_default_library",
        "//vendor/github.com/gogo/protobuf/plugin/embedcheck:go_default_library",
        "//vendor/github.com/gogo/protobuf/plugin/enumstringer:go_default_library",
        "//vendor/github.com/gogo/protobuf/plugin/equal:go_default_library",
        "//vendor/github.com/gogo/protobuf/plugin/face:go_default_library",
        "//vendor/github.com/gogo/protobuf/plugin/gostring:go_default_library",
        "//vendor/github.com/gogo/protobuf/plugin/marshalto:go_default_library",
        "//vendor/github.com/gogo/protobuf/plugin/oneofcheck:go_default_library",
        "//vendor/github.com/gogo/protobuf/plugin/populate:go_default_library",
        "//vendor/github.com/gogo/protobuf/plugin/size:go_default_library",
        "//vendor/github.com/gogo/protobuf/plugin/stringer:go_default_library",
        "//vendor/github.com/gogo/protobuf/plugin/testgen:go_default_library",
        "//vendor/github.com/gogo/protobuf/plugin/union:go_default_library",
        "//vendor/github.com/gogo/protobuf/plugin/unmarshal:go_default_library",
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/github.com/gogo/protobuf/protoc-gen-gogo/generator:go_default_library",
        "//vendor/github.com/gogo/protobuf/protoc-gen-gogo/grpc:go_default_library",
        "//vendor/github.com/gogo/protobuf/protoc-gen-gogo/plugin:go_default_library",
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
