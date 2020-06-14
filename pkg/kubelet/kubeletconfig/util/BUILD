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
        "//pkg/kubelet/kubeletconfig/util/codec:all-srcs",
        "//pkg/kubelet/kubeletconfig/util/equal:all-srcs",
        "//pkg/kubelet/kubeletconfig/util/files:all-srcs",
        "//pkg/kubelet/kubeletconfig/util/log:all-srcs",
        "//pkg/kubelet/kubeletconfig/util/panic:all-srcs",
        "//pkg/kubelet/kubeletconfig/util/test:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
