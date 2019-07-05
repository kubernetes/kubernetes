load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "constants.go",
        "context.go",
        "doc.go",
        "gce_projects.go",
        "gen.go",
        "op.go",
        "project.go",
        "ratelimit.go",
        "service.go",
        "utils.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud",
    importpath = "github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/filter:go_default_library",
        "//vendor/github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/meta:go_default_library",
        "//vendor/google.golang.org/api/compute/v0.alpha:go_default_library",
        "//vendor/google.golang.org/api/compute/v0.beta:go_default_library",
        "//vendor/google.golang.org/api/compute/v1:go_default_library",
        "//vendor/google.golang.org/api/googleapi:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/filter:all-srcs",
        "//vendor/github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/meta:all-srcs",
        "//vendor/github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/mock:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
