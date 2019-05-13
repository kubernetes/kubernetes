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
        "//staging/src/k8s.io/component-base/cli/flag:all-srcs",
        "//staging/src/k8s.io/component-base/cli/globalflag:all-srcs",
        "//staging/src/k8s.io/component-base/config:all-srcs",
        "//staging/src/k8s.io/component-base/featuregate:all-srcs",
        "//staging/src/k8s.io/component-base/logs:all-srcs",
        "//staging/src/k8s.io/component-base/metrics:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
