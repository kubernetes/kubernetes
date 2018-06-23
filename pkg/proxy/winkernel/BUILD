load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "metrics.go",
        "proxier.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/proxy/winkernel",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//pkg/api/service:go_default_library",
            "//pkg/apis/core:go_default_library",
            "//pkg/apis/core/helper:go_default_library",
            "//pkg/proxy:go_default_library",
            "//pkg/proxy/healthcheck:go_default_library",
            "//pkg/util/async:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
            "//staging/src/k8s.io/client-go/tools/record:go_default_library",
            "//vendor/github.com/Microsoft/hcsshim:go_default_library",
            "//vendor/github.com/davecgh/go-spew/spew:go_default_library",
            "//vendor/github.com/golang/glog:go_default_library",
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
