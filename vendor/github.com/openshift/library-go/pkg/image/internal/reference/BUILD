load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "reference.go",
        "regexp.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/library-go/pkg/image/internal/reference",
    importpath = "github.com/openshift/library-go/pkg/image/internal/reference",
    visibility = ["//vendor/github.com/openshift/library-go/pkg/image:__subpackages__"],
    deps = ["//vendor/github.com/openshift/library-go/pkg/image/internal/digest:go_default_library"],
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
