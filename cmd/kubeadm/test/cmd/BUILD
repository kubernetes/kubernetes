package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["util.go"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/test/cmd",
    deps = ["//vendor/github.com/spf13/cobra:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "completion_test.go",
        "init_test.go",
        "join_test.go",
        "token_test.go",
        "version_test.go",
    ],
    args = ["--kubeadm-path=../../kubeadm"],
    data = ["//cmd/kubeadm"],
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/test/cmd",
    tags = [
        "integration",
        "skip",
    ],
    deps = ["//vendor/github.com/ghodss/yaml:go_default_library"],
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
