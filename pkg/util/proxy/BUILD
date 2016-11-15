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
        "dial.go",
        "doc.go",
        "transport.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/net:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//third_party/forked/golang/netutil:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/html",
        "//vendor:golang.org/x/net/html/atom",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "dial_test.go",
        "transport_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = ["//pkg/util/net:go_default_library"],
)
