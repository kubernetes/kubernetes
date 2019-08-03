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
        "//test/images/agnhost:all-srcs",
        "//test/images/apparmor-loader:all-srcs",
        "//test/images/echoserver:all-srcs",
        "//test/images/metadata-concealment:all-srcs",
        "//test/images/nonewprivs:all-srcs",
        "//test/images/pets/peer-finder:all-srcs",
        "//test/images/regression-issue-74839:all-srcs",
        "//test/images/resource-consumer:all-srcs",
        "//test/images/sample-apiserver:all-srcs",
        "//test/images/sample-device-plugin:all-srcs",
    ],
    tags = ["automanaged"],
)
