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
        "config.go",
        "doc.go",
        "keyring.go",
        "plugins.go",
        "provider.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//vendor:github.com/docker/engine-api/types",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "config_test.go",
        "keyring_test.go",
        "provider_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
