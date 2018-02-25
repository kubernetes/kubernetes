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
        "//vendor/github.com/container-storage-interface/spec/lib/go/csi/v0:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
