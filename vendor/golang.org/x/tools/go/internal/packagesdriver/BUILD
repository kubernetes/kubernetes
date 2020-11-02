load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["sizes.go"],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/tools/go/internal/packagesdriver",
    importpath = "golang.org/x/tools/go/internal/packagesdriver",
    visibility = ["//vendor/golang.org/x/tools/go:__subpackages__"],
    deps = ["//vendor/golang.org/x/tools/internal/gocommand:go_default_library"],
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
