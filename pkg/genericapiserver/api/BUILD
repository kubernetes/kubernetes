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
        "apiserver.go",
        "discovery.go",
        "doc.go",
        "groupversion.go",
        "installer.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/api/rest:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/genericapiserver/api/handlers:go_default_library",
        "//pkg/genericapiserver/api/handlers/negotiation:go_default_library",
        "//pkg/genericapiserver/api/handlers/responsewriters:go_default_library",
        "//pkg/genericapiserver/api/metrics:go_default_library",
        "//pkg/genericapiserver/api/request:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/strings:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "apiserver_test.go",
        "installer_test.go",
        "proxy_test.go",
        "watch_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/api/rest:go_default_library",
        "//pkg/api/testing:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/genericapiserver/api/filters:go_default_library",
        "//pkg/genericapiserver/api/handlers:go_default_library",
        "//pkg/genericapiserver/api/handlers/responsewriters:go_default_library",
        "//pkg/genericapiserver/api/request:go_default_library",
        "//pkg/genericapiserver/api/testing:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/runtime/serializer/streaming:go_default_library",
        "//pkg/util/diff:go_default_library",
        "//pkg/util/net:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/watch:go_default_library",
        "//pkg/watch/versioned:go_default_library",
        "//plugin/pkg/admission/admit:go_default_library",
        "//plugin/pkg/admission/deny:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:golang.org/x/net/websocket",
    ],
)
