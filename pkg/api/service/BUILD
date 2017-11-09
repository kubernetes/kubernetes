package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["util.go"],
    importpath = "k8s.io/kubernetes/pkg/api/service",
    deps = [
        "//pkg/apis/core:go_default_library",
        "//pkg/util/net/sets:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["util_test.go"],
    importpath = "k8s.io/kubernetes/pkg/api/service",
    library = ":go_default_library",
    deps = [
        "//pkg/apis/core:go_default_library",
        "//pkg/util/net/sets:go_default_library",
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
