package(default_visibility = ["//visibility:public"])

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
        "//pkg/quota/v1/evaluator/core:all-srcs",
        "//pkg/quota/v1/install:all-srcs",
    ],
    tags = ["automanaged"],
)
