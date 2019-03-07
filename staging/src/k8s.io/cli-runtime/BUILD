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
        "//staging/src/k8s.io/cli-runtime/artifacts:all-srcs",
        "//staging/src/k8s.io/cli-runtime/pkg/genericclioptions:all-srcs",
        "//staging/src/k8s.io/cli-runtime/pkg/kustomize:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
