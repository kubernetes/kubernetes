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
        "//pkg/api/v1:go_default_library",
        "//pkg/api/validation/path:go_default_library",
        "//pkg/client/metrics:go_default_library",
        "//pkg/util/cert:go_default_library",
        "//pkg/version:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/streaming",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/util/flowcontrol",
        "//vendor:k8s.io/client-go/rest/watch",
        "//vendor:k8s.io/client-go/tools/clientcmd/api",
        "//vendor:k8s.io/client-go/transport",
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
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/util/httpstream:go_default_library",
        "//pkg/util/intstr:go_default_library",
        "//pkg/util/testing:go_default_library",
        "//vendor:github.com/google/gofuzz",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/streaming",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/util/clock",
        "//vendor:k8s.io/client-go/pkg/util/flowcontrol",
        "//vendor:k8s.io/client-go/rest/watch",
        "//vendor:k8s.io/client-go/tools/clientcmd/api",
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
    srcs = [
        ":package-srcs",
        "//pkg/client/restclient/fake:all-srcs",
    ],
    tags = ["automanaged"],
)
