package(default_visibility = ["//visibility:public"])

licenses(["notice"])

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
        "//hack/boilerplate:all-srcs",
        "//hack/e2e-internal:all-srcs",
        "//hack/lib:all-srcs",
    ],
    tags = ["automanaged"],
)

sh_test(
    name = "verify-boilerplate",
    srcs = ["verify-boilerplate.sh"],
    data = ["//:all-srcs"],
    tags = ["manual"],
)

sh_test(
    name = "verify-gofmt",
    srcs = ["verify-gofmt.sh"],
    data = [
        "//:all-srcs",
        "@io_bazel_rules_go_toolchain//:toolchain",
    ],
    tags = ["manual"],
)

test_suite(
    name = "verify-all",
    tags = ["manual"],
    tests = [
        "verify-boilerplate",
        "verify-gofmt",
    ],
)
