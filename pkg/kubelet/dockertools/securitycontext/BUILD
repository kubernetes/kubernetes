package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fake.go",
        "provider.go",
        "types.go",
        "util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/leaky:go_default_library",
        "//pkg/securitycontext:go_default_library",
        "//vendor:github.com/docker/engine-api/types/container",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["provider_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/testing:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//vendor:github.com/docker/engine-api/types/container",
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
