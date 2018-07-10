load("@io_bazel_rules_go//go:def.bzl", "go_test")

go_test(
    name = "go_default_test",
    size = "large",
    srcs = ["apps_test.go"],
    importpath = "k8s.io/kubernetes/test/integration/apps",
    tags = [
        "integration",
    ],
    deps = [
        "//test/integration/apps/upgrades:go_default_library",
        "//test/integration/fixtures/controlplane:go_default_library",
        "//test/integration/framework:go_default_library",
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
        "//test/integration/apps/upgrades:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
