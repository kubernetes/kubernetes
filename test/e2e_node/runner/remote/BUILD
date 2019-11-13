package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "remote",
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["run_remote.go"],
    importpath = "k8s.io/kubernetes/test/e2e_node/runner/remote",
    deps = [
        "//test/e2e_node/remote:go_default_library",
        "//test/e2e_node/system:go_default_library",
        "//vendor/github.com/google/uuid:go_default_library",
        "//vendor/golang.org/x/oauth2/google:go_default_library",
        "//vendor/google.golang.org/api/compute/v0.beta:go_default_library",
        "//vendor/google.golang.org/api/option:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
