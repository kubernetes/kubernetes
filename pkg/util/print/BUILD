package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["print_test.go"],
    importpath = "k8s.io/kubernetes/pkg/util/print",
    library = ":go_default_library",
)

go_library(
    name = "go_default_library",
    srcs = ["print.go"],
    importpath = "k8s.io/kubernetes/pkg/util/print",
    deps = ["//vendor/github.com/davecgh/go-spew/spew:go_default_library"],
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
