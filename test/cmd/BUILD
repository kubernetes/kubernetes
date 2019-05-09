sh_library(
    name = "legacy-script",
    srcs = glob(["*.sh"]),
    data = ["//pkg/kubectl/validation:testdata/v1/validPod.yaml"],
    visibility = ["//visibility:public"],
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
