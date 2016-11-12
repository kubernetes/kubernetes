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
        "cluster_helper.go",
        "dns.go",
        "doc.go",
        "endpoint_helper.go",
        "service_helper.go",
        "servicecontroller.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "dns_test.go",
        "endpoint_helper_test.go",
        "service_helper_test.go",
        "servicecontroller_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
