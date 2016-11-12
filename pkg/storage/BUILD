package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "cacher.go",
        "doc.go",
        "errors.go",
        "interfaces.go",
        "selection_predicate.go",
        "util.go",
        "watch_cache.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "selection_predicate_test.go",
        "util_test.go",
        "watch_cache_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)

go_test(
    name = "go_default_xtest",
    srcs = ["cacher_test.go"],
    tags = ["automanaged"],
    deps = [],
)
