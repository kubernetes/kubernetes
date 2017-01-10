package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "generated_expansion.go",
        "storage_client.go",
        "storageclass.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/apis/storage/v1beta1:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/runtime/serializer:go_default_library",
        "//pkg/watch:go_default_library",
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
        "//pkg/client/clientset_generated/clientset/typed/storage/v1beta1/fake:all-srcs",
    ],
    tags = ["automanaged"],
)
