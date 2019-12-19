load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "exitcode.go",
        "flags.go",
        "handler.go",
        "main.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gotest.tools/gotestsum",
    importpath = "gotest.tools/gotestsum",
    visibility = ["//visibility:private"],
    deps = [
        "//vendor/github.com/fatih/color:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/github.com/sirupsen/logrus:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/gotest.tools/gotestsum/internal/junitxml:go_default_library",
        "//vendor/gotest.tools/gotestsum/testjson:go_default_library",
    ],
)

go_binary(
    name = "gotestsum",
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
    srcs = [
        ":package-srcs",
        "//vendor/gotest.tools/gotestsum/internal/junitxml:all-srcs",
        "//vendor/gotest.tools/gotestsum/testjson:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
