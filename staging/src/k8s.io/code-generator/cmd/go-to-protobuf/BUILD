package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "go-to-protobuf",
    importpath = "k8s.io/code-generator/cmd/go-to-protobuf",
    library = ":go_default_library",
)

go_library(
    name = "go_default_library",
    srcs = ["main.go"],
    importpath = "k8s.io/code-generator/cmd/go-to-protobuf",
    deps = [
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/code-generator/cmd/go-to-protobuf/protobuf:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/code-generator/cmd/go-to-protobuf/protobuf:all-srcs",
        "//staging/src/k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo:all-srcs",
    ],
    tags = ["automanaged"],
)
