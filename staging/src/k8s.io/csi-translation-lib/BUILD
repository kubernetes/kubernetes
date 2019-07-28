load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["translate.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/csi-translation-lib",
    importpath = "k8s.io/csi-translation-lib",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/storage/v1:go_default_library",
        "//staging/src/k8s.io/csi-translation-lib/plugins:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["translate_test.go"],
    embed = [":go_default_library"],
    deps = ["//staging/src/k8s.io/api/core/v1:go_default_library"],
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
        "//staging/src/k8s.io/csi-translation-lib/plugins:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
