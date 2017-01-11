package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["eventsink.go"],
    tags = ["automanaged"],
    deps = [
        "//federation/client/clientset_generated/federation_clientset:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/record:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["eventsink_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//federation/client/clientset_generated/federation_clientset/fake:go_default_library",
        "//federation/pkg/federation-controller/util/test:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/testing/core:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
