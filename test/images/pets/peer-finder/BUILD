package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "peer-finder",
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["peer-finder.go"],
    importpath = "k8s.io/kubernetes/test/images/pets/peer-finder",
    deps = ["//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library"],
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
