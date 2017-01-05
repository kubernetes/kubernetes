package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["handlers.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/genericapiserver/api/request:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["handlers_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/genericapiserver/api/request:go_default_library",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)
