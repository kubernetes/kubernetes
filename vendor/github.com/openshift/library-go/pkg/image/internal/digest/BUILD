load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "digest.go",
        "digester.go",
        "doc.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/library-go/pkg/image/internal/digest",
    importpath = "github.com/openshift/library-go/pkg/image/internal/digest",
    visibility = ["//vendor/github.com/openshift/library-go/pkg/image:__subpackages__"],
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
