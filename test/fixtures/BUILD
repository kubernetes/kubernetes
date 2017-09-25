package(default_visibility = ["//visibility:public"])

filegroup(
    name = "fixtures",
    srcs = glob([
        "**/*.yaml",
        "**/*.yml",
        "**/*.json",
    ]),
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
)
