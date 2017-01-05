package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["init_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["initializer.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/controller/informers:go_default_library",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
    ],
)
