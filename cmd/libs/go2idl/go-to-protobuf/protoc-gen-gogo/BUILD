package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_binary(
    name = "protoc-gen-gogo",
    srcs = ["main.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/sortkeys",
        "//vendor:github.com/gogo/protobuf/vanity/command",
    ],
)
