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
        "//pkg/scheduler/framework/plugins/examples/multipoint:all-srcs",
        "//pkg/scheduler/framework/plugins/examples/prebind:all-srcs",
        "//pkg/scheduler/framework/plugins/examples/stateful:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
