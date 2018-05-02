package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["fake.go"],
    importpath = "k8s.io/kubernetes/pkg/util/ipvs/testing",
    deps = ["//pkg/util/ipvs:go_default_library"],
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

go_test(
    name = "go_default_test",
    srcs = ["fake_test.go"],
    embed = [":go_default_library"],
    deps = ["//pkg/util/ipvs:go_default_library"],
)
