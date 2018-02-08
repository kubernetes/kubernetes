package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["tokenfile_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/apiserver/pkg/authentication/token/tokenfile",
    deps = ["//vendor/k8s.io/apiserver/pkg/authentication/user:go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["tokenfile.go"],
    importpath = "k8s.io/apiserver/pkg/authentication/token/tokenfile",
    deps = [
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/user:go_default_library",
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
