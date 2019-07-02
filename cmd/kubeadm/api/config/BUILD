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
        "//cmd/kubeadm/api/config/v1beta1:all-srcs",
        "//cmd/kubeadm/api/config/v1beta2:all-srcs",
    ],
    tags = ["automanaged"],
)
