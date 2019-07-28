load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["go-junit-report.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/jstemmer/go-junit-report",
    importpath = "github.com/jstemmer/go-junit-report",
    visibility = ["//visibility:private"],
    deps = [
        "//vendor/github.com/jstemmer/go-junit-report/formatter:go_default_library",
        "//vendor/github.com/jstemmer/go-junit-report/parser:go_default_library",
    ],
)

go_binary(
    name = "go-junit-report",
    embed = [":go_default_library"],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//vendor/github.com/jstemmer/go-junit-report/formatter:all-srcs",
        "//vendor/github.com/jstemmer/go-junit-report/parser:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
