sh_library(
    name = "lib",
    srcs = [
        "logging.sh",
    ],
    visibility = [
        "//build/visible_to:COMMON_testing",
        "//build/visible_to:cluster",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = [
        "//build/visible_to:cluster",
    ],
)
