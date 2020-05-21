load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["rankedset.go"],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/authentication/oauth/rankedset",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/google/btree:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = ["rankedset_test.go"],
    embed = [":go_default_library"],
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
