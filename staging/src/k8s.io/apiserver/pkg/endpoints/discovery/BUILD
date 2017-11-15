load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "addresses.go",
        "group.go",
        "legacy.go",
        "root.go",
        "util.go",
        "version.go",
    ],
    importpath = "k8s.io/apiserver/pkg/endpoints/discovery",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/endpoints/handlers/negotiation:go_default_library",
        "//pkg/endpoints/handlers/responsewriters:go_default_library",
        "//pkg/endpoints/request:go_default_library",
        "//vendor/github.com/emicklei/go-restful:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/net:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "addresses_test.go",
        "root_test.go",
    ],
    importpath = "k8s.io/apiserver/pkg/endpoints/discovery",
    library = ":go_default_library",
    deps = [
        "//pkg/endpoints/request:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apimachinery/announced:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apimachinery/registered:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/net:go_default_library",
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
    visibility = ["//visibility:public"],
)
