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
        "client.go",
        "config.go",
        "plugin.go",
        "request.go",
        "transport.go",
        "url_utils.go",
        "urlbackoff.go",
        "versions.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/api/validation/path:go_default_library",
        "//pkg/client/metrics:go_default_library",
        "//pkg/client/transport:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/serializer/streaming:go_default_library",
        "//pkg/util/cert:go_default_library",
        "//pkg/util/flowcontrol:go_default_library",
        "//pkg/util/net:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/version:go_default_library",
        "//pkg/watch:go_default_library",
        "//pkg/watch/versioned:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "client_test.go",
        "config_test.go",
        "plugin_test.go",
        "request_test.go",
        "url_utils_test.go",
        "urlbackoff_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/serializer/streaming:go_default_library",
        "//pkg/util/clock:go_default_library",
        "//pkg/util/diff:go_default_library",
        "//pkg/util/flowcontrol:go_default_library",
        "//pkg/util/httpstream:go_default_library",
        "//pkg/util/intstr:go_default_library",
        "//pkg/util/testing:go_default_library",
        "//pkg/watch:go_default_library",
        "//pkg/watch/versioned:go_default_library",
        "//vendor:github.com/google/gofuzz",
    ],
)
