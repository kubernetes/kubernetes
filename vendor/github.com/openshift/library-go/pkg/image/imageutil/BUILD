load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["helpers.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/library-go/pkg/image/imageutil",
    importpath = "github.com/openshift/library-go/pkg/image/imageutil",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/blang/semver:go_default_library",
        "//vendor/github.com/openshift/api/image/docker10:go_default_library",
        "//vendor/github.com/openshift/api/image/v1:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/image/internal/digest:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/image/reference:go_default_library",
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
