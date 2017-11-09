package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["fields.go"],
    importpath = "k8s.io/apimachinery/third_party/forked/golang/json",
)

go_test(
    name = "go_default_test",
    srcs = ["fields_test.go"],
    importpath = "k8s.io/apimachinery/third_party/forked/golang/json",
    library = ":go_default_library",
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
