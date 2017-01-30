package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
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
        "//third_party/forked/golang/netutil:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/html",
        "//vendor:golang.org/x/net/html/atom",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "dial_test.go",
        "transport_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/util/net"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
