load("@io_bazel_rules_go//go:def.bzl", "go_test")

go_test(
    name = "go_default_test",
    srcs = ["PrintWorkDir_test.go"],
    tags = ["manual"],
    deps = [
        "//vendor/sigs.k8s.io/kustomize/pkg/kusttest:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/plugin:go_default_library",
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
    visibility = ["//visibility:public"],
)
