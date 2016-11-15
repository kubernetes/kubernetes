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
        "api.go",
        "doc.go",
        "healthcheck.go",
        "http.go",
        "listener.go",
        "worker.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/client/cache:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["healthcheck_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/types:go_default_library",
        "//pkg/util/sets:go_default_library",
    ],
)
