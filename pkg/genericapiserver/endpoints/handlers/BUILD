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
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/genericapiserver/registry/rest:go_default_library",
        "//pkg/util/strategicpatch:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/evanphx/json-patch",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "discovery.go",
        "doc.go",
        "patch.go",
        "proxy.go",
        "rest.go",
        "watch.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/genericapiserver/endpoints/handlers/responsewriters:go_default_library",
        "//pkg/genericapiserver/registry/rest:go_default_library",
        "//pkg/util:go_default_library",
        "//pkg/util/httpstream:go_default_library",
        "//pkg/util/proxy:go_default_library",
        "//pkg/util/strategicpatch:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/evanphx/json-patch",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/websocket",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/internalversion",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/streaming",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apiserver/pkg/admission",
        "//vendor:k8s.io/apiserver/pkg/endpoints/handlers/negotiation",
        "//vendor:k8s.io/apiserver/pkg/endpoints/metrics",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
        "//vendor:k8s.io/apiserver/pkg/server/httplog",
        "//vendor:k8s.io/apiserver/pkg/util/wsstream",
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
        "//pkg/genericapiserver/endpoints/handlers/responsewriters:all-srcs",
    ],
    tags = ["automanaged"],
)
