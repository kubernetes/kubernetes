package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "error.go",
        "template.go",
        "version.go",
    ],
    tags = ["automanaged"],
    deps = ["//cmd/kubeadm/app/preflight:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "error_test.go",
        "template_test.go",
        "version_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//cmd/kubeadm/app/preflight:go_default_library"],
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
        "//cmd/kubeadm/app/util/kubeconfig:all-srcs",
        "//cmd/kubeadm/app/util/token:all-srcs",
    ],
    tags = ["automanaged"],
)
