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
        "//staging/src/k8s.io/cri-api/pkg/apis:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
