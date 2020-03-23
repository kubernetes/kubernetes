load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "key.go",
        "meta.go",
        "method.go",
        "service.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/meta",
    importpath = "github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/meta",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/google.golang.org/api/compute/v0.alpha:go_default_library",
        "//vendor/google.golang.org/api/compute/v0.beta:go_default_library",
        "//vendor/google.golang.org/api/compute/v1:go_default_library",
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
