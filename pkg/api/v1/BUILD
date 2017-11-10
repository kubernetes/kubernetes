load("@io_bazel_rules_go//go:def.bzl", "go_test")

go_test(
    name = "go_default_xtest",
    srcs = ["generate_test.go"],
    importpath = "k8s.io/kubernetes/pkg/api/v1_test",
    deps = [
        ":go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//pkg/api/v1/endpoints:all-srcs",
        "//pkg/api/v1/node:all-srcs",
        "//pkg/api/v1/pod:all-srcs",
        "//pkg/api/v1/resource:all-srcs",
        "//pkg/api/v1/service:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
