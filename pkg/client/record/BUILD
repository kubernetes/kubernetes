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
    deps = [
        "//pkg/api/errors:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/clock:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/strategicpatch:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/golang/groupcache/lru",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "event_test.go",
        "events_cache_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/errors:go_default_library",
        "//pkg/api/install:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/clock:go_default_library",
        "//pkg/util/diff:go_default_library",
        "//pkg/util/strategicpatch:go_default_library",
    ],
)
