package(default_visibility = ["//visibility:public"])

filegroup(
    name = "sources",
    srcs = glob([
        "**/*",
    ]),
)

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
        "//test/images/clusterapi-tester:all-srcs",
        "//test/images/entrypoint-tester:all-srcs",
        "//test/images/fakegitserver:all-srcs",
        "//test/images/goproxy:all-srcs",
        "//test/images/liveness:all-srcs",
        "//test/images/logs-generator:all-srcs",
        "//test/images/mounttest:all-srcs",
        "//test/images/n-way-http:all-srcs",
        "//test/images/net:all-srcs",
        "//test/images/netexec:all-srcs",
        "//test/images/nettest:all-srcs",
        "//test/images/no-snat-test:all-srcs",
        "//test/images/no-snat-test-proxy:all-srcs",
        "//test/images/nonewprivs:all-srcs",
        "//test/images/pets/peer-finder:all-srcs",
        "//test/images/port-forward-tester:all-srcs",
        "//test/images/porter:all-srcs",
        "//test/images/resource-consumer:all-srcs",
        "//test/images/serve-hostname:all-srcs",
        "//test/images/test-webserver:all-srcs",
    ],
    tags = ["automanaged"],
)
