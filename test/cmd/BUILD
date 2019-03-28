sh_library(
    name = "legacy-script",
    srcs = glob(["*.sh"]),
    data = ["//staging/src/k8s.io/kubectl/pkg/validation:testdata/v1/validPod.yaml"],
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
