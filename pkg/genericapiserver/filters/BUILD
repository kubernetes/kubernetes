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
        "cors.go",
        "doc.go",
        "longrunning.go",
        "maxinflight.go",
        "panics.go",
        "timeout.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/apiserver/request:go_default_library",
        "//pkg/httplog:go_default_library",
        "//pkg/util:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "cors_test.go",
        "maxinflight_test.go",
        "timeout_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/apiserver/filters:go_default_library",
        "//pkg/apiserver/request:go_default_library",
        "//pkg/util/sets:go_default_library",
    ],
)
