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
        "discovery_client.go",
        "helper.go",
        "restmapper.go",
        "unstructured.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/serializer:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/version:go_default_library",
        "//plugin/pkg/client/auth:go_default_library",
        "//vendor:github.com/emicklei/go-restful/swagger",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "discovery_client_test.go",
        "restmapper_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/errors:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/restclient/fake:go_default_library",
        "//pkg/version:go_default_library",
        "//vendor:github.com/emicklei/go-restful/swagger",
        "//vendor:github.com/stretchr/testify/assert",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["helper_blackbox_test.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/restclient/fake:go_default_library",
        "//pkg/client/typed/discovery:go_default_library",
        "//pkg/runtime:go_default_library",
    ],
)
