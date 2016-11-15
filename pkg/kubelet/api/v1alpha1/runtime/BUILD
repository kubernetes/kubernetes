package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "api.pb.go",
        "constants.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:golang.org/x/net/context",
        "//vendor:google.golang.org/grpc",
    ],
)
