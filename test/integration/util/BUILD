package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "cloud.go",
        "util.go",
    ],
    importpath = "k8s.io/kubernetes/test/integration/util",
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/scheduler:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/informers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/tools/events:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/gce:go_default_library",
        "//test/integration/framework:go_default_library",
        "//vendor/github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud:go_default_library",
        "//vendor/golang.org/x/oauth2:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
