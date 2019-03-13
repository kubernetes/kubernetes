load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "container.go",
        "util.go",
        "vm.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/Microsoft/hcsshim/internal/runhcs",
    importpath = "github.com/Microsoft/hcsshim/internal/runhcs",
    visibility = ["//vendor/github.com/Microsoft/hcsshim:__subpackages__"],
    deps = [
        "//vendor/github.com/Microsoft/go-winio:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/guid:go_default_library",
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
