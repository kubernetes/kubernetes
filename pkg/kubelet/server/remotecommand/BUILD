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
        "attach.go",
        "constants.go",
        "doc.go",
        "exec.go",
        "httpstream.go",
        "websocket.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/httplog:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/httpstream:go_default_library",
        "//pkg/util/httpstream/spdy:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/term:go_default_library",
        "//pkg/util/wsstream:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
