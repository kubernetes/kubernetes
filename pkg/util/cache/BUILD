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
        "cache.go",
        "lruexpirecache.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/groupcache/lru"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "cache_test.go",
        "lruexpirecache_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/groupcache/lru"],
)
