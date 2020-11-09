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
        "//staging/src/k8s.io/component-helpers/auth/rbac/reconciliation:all-srcs",
        "//staging/src/k8s.io/component-helpers/auth/rbac/validation:all-srcs",
        "//staging/src/k8s.io/component-helpers/lease:all-srcs",
        "//staging/src/k8s.io/component-helpers/scheduling/corev1:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
