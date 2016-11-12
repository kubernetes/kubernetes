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
        "doc.go",
        "event.go",
        "events_cache.go",
        "fake.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "event_test.go",
        "events_cache_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
