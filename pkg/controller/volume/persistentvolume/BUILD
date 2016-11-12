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
        "index.go",
        "pv_controller.go",
        "pv_controller_base.go",
        "volume_host.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "binder_test.go",
        "delete_test.go",
        "framework_test.go",
        "index_test.go",
        "provision_test.go",
        "pv_controller_test.go",
        "recycle_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
