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
        "log.go",
        "subresources.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/rest:go_default_library",
        "//pkg/api/validation:go_default_library",
        "//pkg/capabilities:go_default_library",
        "//pkg/kubelet/client:go_default_library",
        "//pkg/registry/core/pod:go_default_library",
        "//pkg/registry/generic/registry:go_default_library",
        "//pkg/registry/generic/rest:go_default_library",
        "//pkg/runtime:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["log_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/registry/generic:go_default_library",
        "//pkg/registry/generic/registry:go_default_library",
        "//pkg/registry/registrytest:go_default_library",
    ],
)
