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
        "client_builder.go",
        "controller_ref_manager.go",
        "controller_utils.go",
        "doc.go",
        "lookup_cache.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/api/validation:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/core/internalversion:go_default_library",
        "//pkg/client/record:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/serviceaccount:go_default_library",
        "//pkg/util/clock:go_default_library",
        "//pkg/util/hash:go_default_library",
        "//pkg/util/integer:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/golang/groupcache/lru",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["controller_utils_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/record:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/securitycontext:go_default_library",
        "//pkg/util/clock:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/testing:go_default_library",
        "//pkg/util/uuid:go_default_library",
    ],
)
