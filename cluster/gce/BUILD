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
        "//cluster/gce/addons:all-srcs",
        "//cluster/gce/gci:all-srcs",
        "//cluster/gce/manifests:all-srcs",
        "//cluster/gce/windows:all-srcs",
    ],
    tags = ["automanaged"],
)
