load("@io_bazel_rules_go//go:def.bzl", "go_test")

go_test(
    name = "go_default_test",
    srcs = ["apps_test.go"],
    deps = [
        "//test/integration/apps/common_types:go_default_library",
        "//test/integration/apps/core:go_default_library",
        "//test/integration/apps/hrcontrollers:go_default_library",
        "//test/integration/apps/podconfig:go_default_library",
        "//test/integration/apps/podcontrollers:go_default_library",
        "//test/integration/apps/upgrades:go_default_library",
        "//test/integration/framework:go_default_library",
    ],
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
        "//test/integration/apps/common_types:all-srcs",
        "//test/integration/apps/core:all-srcs",
        "//test/integration/apps/hrcontrollers:all-srcs",
        "//test/integration/apps/podconfig:all-srcs",
        "//test/integration/apps/podcontrollers:all-srcs",
        "//test/integration/apps/upgrades:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
