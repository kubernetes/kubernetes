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
        "api_object_versioner.go",
        "doc.go",
        "etcd_helper.go",
        "etcd_watcher.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "api_object_versioner_test.go",
        "etcd_helper_test.go",
        "etcd_watcher_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
