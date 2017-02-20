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
        "default_storage_factory_builder.go",
        "doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apis/pkg/api",
        "//vendor:k8s.io/apiserver/pkg/server/storage",
        "//vendor:k8s.io/apiserver/pkg/storage/storagebackend",
        "//vendor:k8s.io/apiserver/pkg/util/flag",
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
        "//pkg/kubeapiserver/admission:all-srcs",
        "//pkg/kubeapiserver/authenticator:all-srcs",
        "//pkg/kubeapiserver/authorizer:all-srcs",
        "//pkg/kubeapiserver/options:all-srcs",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = ["default_storage_factory_builder_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apis/pkg/api",
        "//vendor:k8s.io/apis/pkg/api/install",
        "//vendor:k8s.io/apis/pkg/api/v1",
        "//vendor:k8s.io/apis/pkg/apis/extensions/install",
        "//vendor:k8s.io/apis/pkg/apis/extensions/v1beta1",
        "//vendor:k8s.io/apiserver/pkg/server/storage",
    ],
)
