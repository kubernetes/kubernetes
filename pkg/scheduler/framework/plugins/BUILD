load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["default_registry.go"],
    importpath = "k8s.io/kubernetes/pkg/scheduler/framework/plugins",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/scheduler/algorithm:go_default_library",
        "//pkg/scheduler/algorithm/predicates:go_default_library",
        "//pkg/scheduler/algorithm/priorities:go_default_library",
        "//pkg/scheduler/apis/config:go_default_library",
        "//pkg/scheduler/framework/plugins/imagelocality:go_default_library",
        "//pkg/scheduler/framework/plugins/nodeaffinity:go_default_library",
        "//pkg/scheduler/framework/plugins/nodename:go_default_library",
        "//pkg/scheduler/framework/plugins/nodeports:go_default_library",
        "//pkg/scheduler/framework/plugins/nodepreferavoidpods:go_default_library",
        "//pkg/scheduler/framework/plugins/noderesources:go_default_library",
        "//pkg/scheduler/framework/plugins/tainttoleration:go_default_library",
        "//pkg/scheduler/framework/plugins/volumebinding:go_default_library",
        "//pkg/scheduler/framework/plugins/volumerestrictions:go_default_library",
        "//pkg/scheduler/framework/plugins/volumezone:go_default_library",
        "//pkg/scheduler/framework/v1alpha1:go_default_library",
        "//pkg/scheduler/internal/cache:go_default_library",
        "//pkg/scheduler/volumebinder:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/client-go/listers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/listers/storage/v1:go_default_library",
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
        "//pkg/scheduler/framework/plugins/examples:all-srcs",
        "//pkg/scheduler/framework/plugins/imagelocality:all-srcs",
        "//pkg/scheduler/framework/plugins/migration:all-srcs",
        "//pkg/scheduler/framework/plugins/nodeaffinity:all-srcs",
        "//pkg/scheduler/framework/plugins/nodename:all-srcs",
        "//pkg/scheduler/framework/plugins/nodeports:all-srcs",
        "//pkg/scheduler/framework/plugins/nodepreferavoidpods:all-srcs",
        "//pkg/scheduler/framework/plugins/noderesources:all-srcs",
        "//pkg/scheduler/framework/plugins/tainttoleration:all-srcs",
        "//pkg/scheduler/framework/plugins/volumebinding:all-srcs",
        "//pkg/scheduler/framework/plugins/volumerestrictions:all-srcs",
        "//pkg/scheduler/framework/plugins/volumezone:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["default_registry_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/scheduler/apis/config:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp:go_default_library",
    ],
)
