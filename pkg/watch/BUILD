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
        "doc.go",
        "filter.go",
        "mux.go",
        "streamwatcher.go",
        "until.go",
        "watch.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/util/net:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = [
        "filter_test.go",
        "mux_test.go",
        "streamwatcher_test.go",
        "watch_test.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/watch:go_default_library",
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
        "//pkg/watch/json:all-srcs",
        "//pkg/watch/versioned:all-srcs",
    ],
    tags = ["automanaged"],
)
