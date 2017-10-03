filegroup(
    name = "testdata",
    srcs = [
        "example/instances/exampleinstance/Kube-manifest.yaml",
        "example/instances/exampleinstance/deployment.yaml",
        "example/package/Kube-manifest.yaml",
        "example/package/deployment.yaml",
    ],
    visibility = ["//build/visible_to:COMMON_testing"],
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
