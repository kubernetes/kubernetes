package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["goroutinemap.go"],
    tags = ["automanaged"],
    deps = ["//pkg/util/goroutinemap/exponentialbackoff:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = ["goroutinemap_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
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
        "//pkg/util/goroutinemap/exponentialbackoff:all-srcs",
    ],
    tags = ["automanaged"],
)
