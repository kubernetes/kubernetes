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
        "//staging/src/k8s.io/sample-cli-plugin/cmd:all-srcs",
        "//staging/src/k8s.io/sample-cli-plugin/pkg/cmd:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
