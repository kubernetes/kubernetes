load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["staticcheck.go"],
    importmap = "k8s.io/kubernetes/vendor/honnef.co/go/tools/cmd/staticcheck",
    importpath = "honnef.co/go/tools/cmd/staticcheck",
    visibility = ["//visibility:private"],
    deps = [
        "//vendor/golang.org/x/tools/go/analysis:go_default_library",
        "//vendor/honnef.co/go/tools/lint:go_default_library",
        "//vendor/honnef.co/go/tools/lint/lintutil:go_default_library",
        "//vendor/honnef.co/go/tools/simple:go_default_library",
        "//vendor/honnef.co/go/tools/staticcheck:go_default_library",
        "//vendor/honnef.co/go/tools/stylecheck:go_default_library",
        "//vendor/honnef.co/go/tools/unused:go_default_library",
    ],
)

go_binary(
    name = "staticcheck",
    embed = [":go_default_library"],
    tags = ["manual"],
    visibility = ["//visibility:public"],
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
