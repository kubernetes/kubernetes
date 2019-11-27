load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "execution.go",
        "format.go",
        "pkgpathprefix.go",
        "summary.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gotest.tools/gotestsum/testjson",
    importpath = "gotest.tools/gotestsum/testjson",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/fatih/color:go_default_library",
        "//vendor/github.com/jonboulle/clockwork:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/github.com/sirupsen/logrus:go_default_library",
        "//vendor/golang.org/x/sync/errgroup:go_default_library",
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
