load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

licenses(["notice"])

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "glog.go",
        "glog_file.go",
    ],
    importpath = "k8s.io/kubernetes/third_party/forked/glog",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["glog_test.go"],
    embed = [":go_default_library"],
)
