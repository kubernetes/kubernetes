package(default_visibility = ["//visibility:public"])

sh_library(
    name = "lib",
    srcs = [
        "etcd.sh",
        "golang.sh",
        "init.sh",
        "logging.sh",
        "swagger.sh",
        "test.sh",
        "util.sh",
        "version.sh",
    ],
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
