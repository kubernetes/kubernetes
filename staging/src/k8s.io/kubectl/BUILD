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
        "//staging/src/k8s.io/kubectl/pkg/util/deployment:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/hash:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/interrupt:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/podutils:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/printers:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/rbac:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/templates:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/term:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
