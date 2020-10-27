package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "connection_test.go",
        "roundtripper_test.go",
        "upgrade_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/httpstream:go_default_library",
        "//vendor/github.com/docker/spdystream:go_default_library",
        "//vendor/github.com/elazarl/goproxy:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "connection.go",
        "roundtripper.go",
        "upgrade.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/util/httpstream/spdy",
    importpath = "k8s.io/apimachinery/pkg/util/httpstream/spdy",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/httpstream:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/net:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/third_party/forked/golang/netutil:go_default_library",
        "//vendor/github.com/docker/spdystream:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
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
