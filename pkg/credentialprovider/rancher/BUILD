package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["rancher_registry_credentials_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/credentialprovider:go_default_library",
        "//vendor/github.com/rancher/go-rancher/client:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "rancher_registry_credentials.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/credentialprovider/rancher",
    deps = [
        "//pkg/credentialprovider:go_default_library",
        "//vendor/github.com/rancher/go-rancher/client:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
