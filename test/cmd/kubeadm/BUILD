package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["util.go"],
    importpath = "k8s.io/kubernetes/test/cmd/kubeadm",
    deps = ["//vendor/github.com/pkg/errors:go_default_library"],
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
    data = glob(["testdata/**"]),
    embed = [":go_default_library"],
    rundir = ".",
    tags = ["integration"],
    deps = [
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/phases/certs:go_default_library",
        "//cmd/kubeadm/app/testing:go_default_library",
        "//cmd/kubeadm/app/util/pkiutil:go_default_library",
        "//vendor/github.com/lithammer/dedent:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/sigs.k8s.io/yaml:go_default_library",
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
