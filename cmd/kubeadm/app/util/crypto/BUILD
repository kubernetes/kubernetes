load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["crypto.go"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/util/crypto",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/pkg/errors:go_default_library"],
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
    srcs = ["crypto_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//cmd/kubeadm/app/constants:go_default_library",
        "//vendor/github.com/lithammer/dedent:go_default_library",
    ],
)
