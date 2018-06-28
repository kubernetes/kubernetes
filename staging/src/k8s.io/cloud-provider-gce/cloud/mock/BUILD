load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["mock.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/cloud-provider-gce/cloud/mock",
    importpath = "k8s.io/cloud-provider-gce/cloud/mock",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/cloud-provider-gce/cloud:go_default_library",
        "//staging/src/k8s.io/cloud-provider-gce/cloud/filter:go_default_library",
        "//staging/src/k8s.io/cloud-provider-gce/cloud/meta:go_default_library",
        "//vendor/google.golang.org/api/compute/v0.alpha:go_default_library",
        "//vendor/google.golang.org/api/compute/v0.beta:go_default_library",
        "//vendor/google.golang.org/api/compute/v1:go_default_library",
        "//vendor/google.golang.org/api/googleapi:go_default_library",
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
