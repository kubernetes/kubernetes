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
        "//cmd/kubeadm/app/phases/addons/dns:all-srcs",
        "//cmd/kubeadm/app/phases/addons/proxy:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
