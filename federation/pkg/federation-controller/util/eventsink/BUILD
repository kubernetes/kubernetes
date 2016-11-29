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
    srcs = ["eventsink.go"],
    tags = ["automanaged"],
    deps = [
        "//federation/client/clientset_generated/federation_release_1_5:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/record:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["eventsink_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//federation/client/clientset_generated/federation_release_1_5/fake:go_default_library",
        "//federation/pkg/federation-controller/util/test:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/testing/core:go_default_library",
        "//pkg/runtime:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
    ],
)
