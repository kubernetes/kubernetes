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
    name = "network-tester",
    srcs = ["webserver.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/unversioned:go_default_library",
        "//pkg/util/sets:go_default_library",
    ],
)
