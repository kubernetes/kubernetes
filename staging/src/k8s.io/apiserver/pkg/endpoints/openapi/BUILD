package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["openapi_test.go"],
    importpath = "k8s.io/apiserver/pkg/endpoints/openapi",
    library = ":go_default_library",
    deps = [
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/endpoints/openapi/testing:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["openapi.go"],
    importpath = "k8s.io/apiserver/pkg/endpoints/openapi",
    deps = [
        "//vendor/github.com/emicklei/go-restful:go_default_library",
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/util:go_default_library",
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
        "//staging/src/k8s.io/apiserver/pkg/endpoints/openapi/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
