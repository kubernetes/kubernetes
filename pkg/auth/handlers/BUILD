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
    srcs = ["handlers.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/auth/authenticator:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/prometheus/client_golang/prometheus",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["handlers_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/auth/authenticator:go_default_library",
        "//pkg/auth/user:go_default_library",
    ],
)
