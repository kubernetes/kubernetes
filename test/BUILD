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
        "//test/cmd:all-srcs",
        "//test/conformance:all-srcs",
        "//test/e2e:all-srcs",
        "//test/e2e_kubeadm:all-srcs",
        "//test/e2e_node:all-srcs",
        "//test/fixtures:all-srcs",
        "//test/images:all-srcs",
        "//test/integration:all-srcs",
        "//test/kubemark:all-srcs",
        "//test/list:all-srcs",
        "//test/soak/cauldron:all-srcs",
        "//test/soak/serve_hostnames:all-srcs",
        "//test/typecheck:all-srcs",
        "//test/utils:all-srcs",
    ],
    tags = ["automanaged"],
)
