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
        "//staging/src/k8s.io/metrics/hack:all-srcs",
        "//staging/src/k8s.io/metrics/pkg/apis/custom_metrics:all-srcs",
        "//staging/src/k8s.io/metrics/pkg/apis/external_metrics:all-srcs",
        "//staging/src/k8s.io/metrics/pkg/apis/metrics:all-srcs",
        "//staging/src/k8s.io/metrics/pkg/client/clientset/versioned:all-srcs",
        "//staging/src/k8s.io/metrics/pkg/client/clientset_test:all-srcs",
        "//staging/src/k8s.io/metrics/pkg/client/custom_metrics:all-srcs",
        "//staging/src/k8s.io/metrics/pkg/client/external_metrics:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
