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
        "admission.go",
        "interfaces.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/listers/core/internalversion:go_default_library",
        "//pkg/controller/informers:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//vendor:github.com/hashicorp/golang-lru",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["admission_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/fake:go_default_library",
        "//pkg/client/testing/core:go_default_library",
        "//pkg/controller/informers:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/wait:go_default_library",
    ],
)
