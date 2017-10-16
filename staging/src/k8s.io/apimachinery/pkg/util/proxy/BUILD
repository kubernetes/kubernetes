package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "dial_test.go",
        "transport_test.go",
        "upgradeaware_test.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/util/proxy",
    library = ":go_default_library",
    deps = [
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
        "//vendor/golang.org/x/net/websocket:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/httpstream:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/net:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "dial.go",
        "doc.go",
        "transport.go",
        "upgradeaware.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/util/proxy",
    deps = [
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/mxk/go-flowrate/flowrate:go_default_library",
        "//vendor/golang.org/x/net/html:go_default_library",
        "//vendor/golang.org/x/net/html/atom:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/httpstream:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/net:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/third_party/forked/golang/netutil:go_default_library",
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
