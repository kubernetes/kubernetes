package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "default_storage_factory_builder.go",
        "doc.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubeapiserver",
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/apis/apps:go_default_library",
        "//pkg/apis/batch:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//pkg/apis/events:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/networking:go_default_library",
        "//pkg/apis/policy:go_default_library",
        "//pkg/apis/storage:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/options:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/options/encryptionconfig:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/resourceconfig:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/storage:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/storagebackend:go_default_library",
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
        "//pkg/kubeapiserver/server:all-srcs",
    ],
    tags = ["automanaged"],
)
