package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
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
        "//pkg/api:go_default_library",
        "//pkg/api/testing:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/genericapiserver/endpoints/filters:go_default_library",
        "//pkg/genericapiserver/endpoints/handlers:go_default_library",
        "//pkg/genericapiserver/endpoints/handlers/responsewriters:go_default_library",
        "//pkg/genericapiserver/endpoints/testing:go_default_library",
        "//pkg/genericapiserver/registry/rest:go_default_library",
        "//plugin/pkg/admission/admit:go_default_library",
        "//plugin/pkg/admission/deny:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:golang.org/x/net/websocket",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/internalversion",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/streaming",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apiserver/pkg/admission",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
    ],
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
        "//pkg/apis/extensions:go_default_library",
        "//pkg/genericapiserver/endpoints/handlers:go_default_library",
        "//pkg/genericapiserver/endpoints/handlers/responsewriters:go_default_library",
        "//pkg/genericapiserver/registry/rest:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apiserver/pkg/admission",
        "//vendor:k8s.io/apiserver/pkg/endpoints/handlers/negotiation",
        "//vendor:k8s.io/apiserver/pkg/endpoints/metrics",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
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
        "//pkg/genericapiserver/endpoints/filters:all-srcs",
        "//pkg/genericapiserver/endpoints/handlers:all-srcs",
        "//pkg/genericapiserver/endpoints/openapi:all-srcs",
        "//pkg/genericapiserver/endpoints/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
