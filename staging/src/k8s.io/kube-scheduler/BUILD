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
        "//staging/src/k8s.io/kube-scheduler/config/v1:all-srcs",
        "//staging/src/k8s.io/kube-scheduler/config/v1alpha2:all-srcs",
        "//staging/src/k8s.io/kube-scheduler/extender/v1:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
