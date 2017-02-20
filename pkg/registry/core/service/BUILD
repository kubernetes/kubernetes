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
        "doc.go",
        "proxy.go",
        "registry.go",
        "rest.go",
        "strategy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/service:go_default_library",
        "//pkg/api/validation:go_default_library",
        "//pkg/capabilities:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/registry/core/endpoint:go_default_library",
        "//pkg/registry/core/service/ipallocator:go_default_library",
        "//pkg/registry/core/service/portallocator:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "rest_test.go",
        "strategy_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/service:go_default_library",
        "//pkg/api/testing:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/registry/core/service/ipallocator:go_default_library",
        "//pkg/registry/core/service/portallocator:go_default_library",
        "//pkg/registry/registrytest:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
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
        "//pkg/registry/core/service/allocator:all-srcs",
        "//pkg/registry/core/service/ipallocator:all-srcs",
        "//pkg/registry/core/service/portallocator:all-srcs",
        "//pkg/registry/core/service/storage:all-srcs",
    ],
    tags = ["automanaged"],
)
