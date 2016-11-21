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
        "proxy.go",
        "response_checker.go",
        "streamer.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/errors:go_default_library",
        "//pkg/api/rest:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/util/config:go_default_library",
        "//pkg/util/httpstream:go_default_library",
        "//pkg/util/net:go_default_library",
        "//pkg/util/proxy:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/mxk/go-flowrate/flowrate",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "proxy_test.go",
        "response_checker_test.go",
        "streamer_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/util/config:go_default_library",
        "//pkg/util/httpstream:go_default_library",
        "//pkg/util/net:go_default_library",
        "//pkg/util/proxy:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:github.com/stretchr/testify/require",
        "//vendor:golang.org/x/net/websocket",
    ],
)
