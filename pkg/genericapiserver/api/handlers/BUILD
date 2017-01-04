package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["rest_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/rest:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/genericapiserver/api/request:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/diff:go_default_library",
        "//pkg/util/strategicpatch:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/evanphx/json-patch",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "discovery.go",
        "doc.go",
        "proxy.go",
        "rest.go",
        "watch.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/api/rest:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/genericapiserver/api/handlers/negotiation:go_default_library",
        "//pkg/genericapiserver/api/handlers/responsewriters:go_default_library",
        "//pkg/genericapiserver/api/metrics:go_default_library",
        "//pkg/genericapiserver/api/request:go_default_library",
        "//pkg/httplog:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/runtime/serializer/streaming:go_default_library",
        "//pkg/util:go_default_library",
        "//pkg/util/httpstream:go_default_library",
        "//pkg/util/net:go_default_library",
        "//pkg/util/proxy:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/strategicpatch:go_default_library",
        "//pkg/util/wsstream:go_default_library",
        "//pkg/watch:go_default_library",
        "//pkg/watch/versioned:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/evanphx/json-patch",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/websocket",
    ],
)
