load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "isatty_bsd.go",
        "isatty_linux.go",
        "isatty_solaris.go",
        "isatty_windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/onsi/ginkgo/reporters/stenographer/support/go-isatty",
    importpath = "github.com/onsi/ginkgo/reporters/stenographer/support/go-isatty",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
