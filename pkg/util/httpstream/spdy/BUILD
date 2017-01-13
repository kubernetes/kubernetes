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
        "connection.go",
        "roundtripper.go",
        "upgrade.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/util/httpstream:go_default_library",
        "//third_party/forked/golang/netutil:go_default_library",
        "//vendor:github.com/docker/spdystream",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "connection_test.go",
        "roundtripper_test.go",
        "upgrade_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/util/httpstream:go_default_library",
        "//vendor:github.com/elazarl/goproxy",
    ],
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
