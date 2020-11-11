load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["report.go"],
    importmap = "k8s.io/kubernetes/vendor/honnef.co/go/tools/report",
    importpath = "honnef.co/go/tools/report",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/tools/go/analysis:go_default_library",
        "//vendor/golang.org/x/tools/go/ast/astutil:go_default_library",
        "//vendor/honnef.co/go/tools/facts:go_default_library",
        "//vendor/honnef.co/go/tools/lint:go_default_library",
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
    visibility = ["//visibility:public"],
)
