load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "project.go",
        "quota.go",
        "quota_linux.go",
        "quota_unsupported.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/volume/util/fsquota",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/features:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//vendor/k8s.io/utils/mount:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//pkg/volume/util/fsquota/common:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/volume/util/fsquota/common:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = ["quota_linux_test.go"],
    embed = [":go_default_library"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//pkg/features:go_default_library",
            "//pkg/volume/util/fsquota/common:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
            "//staging/src/k8s.io/component-base/featuregate/testing:go_default_library",
            "//vendor/k8s.io/utils/mount:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/features:go_default_library",
            "//pkg/volume/util/fsquota/common:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
            "//staging/src/k8s.io/component-base/featuregate/testing:go_default_library",
            "//vendor/k8s.io/utils/mount:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
        "//pkg/volume/util/fsquota/common:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
