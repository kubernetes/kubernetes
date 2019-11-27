load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["lint.go"],
    importmap = "k8s.io/kubernetes/vendor/honnef.co/go/tools/internal/sharedcheck",
    importpath = "honnef.co/go/tools/internal/sharedcheck",
    visibility = ["//vendor/honnef.co/go/tools:__subpackages__"],
    deps = [
        "//vendor/golang.org/x/tools/go/analysis:go_default_library",
        "//vendor/honnef.co/go/tools/internal/passes/buildssa:go_default_library",
        "//vendor/honnef.co/go/tools/lint/lintdsl:go_default_library",
        "//vendor/honnef.co/go/tools/ssa:go_default_library",
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
