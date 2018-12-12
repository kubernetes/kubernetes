load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["interface.go"],
    importpath = "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion/coordination",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/client/informers/informers_generated/internalversion/coordination/internalversion:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion/internalinterfaces:go_default_library",
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
        "//pkg/client/informers/informers_generated/internalversion/coordination/internalversion:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
