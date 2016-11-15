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
        "conn.go",
        "doc.go",
        "stream.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/runtime:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/websocket",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "conn_test.go",
        "stream_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/net/websocket"],
)
