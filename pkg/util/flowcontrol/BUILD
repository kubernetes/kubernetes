package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "backoff.go",
        "throttle.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/clock:go_default_library",
        "//pkg/util/integer:go_default_library",
        "//vendor:github.com/juju/ratelimit",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "backoff_test.go",
        "throttle_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//pkg/util/clock:go_default_library"],
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
