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
        "//pkg/printers/internalversion:all-srcs",
        "//pkg/printers/storage:all-srcs",
    ],
    tags = ["automanaged"],
)
