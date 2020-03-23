package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["registry.go"],
    importpath = "k8s.io/kubernetes/pkg/scheduler/algorithmprovider",
    deps = [
        "//pkg/features:go_default_library",
        "//pkg/scheduler/apis/config:go_default_library",
        "//pkg/scheduler/framework/plugins/defaultbinder:go_default_library",
        "//pkg/scheduler/framework/plugins/defaultpodtopologyspread:go_default_library",
        "//pkg/scheduler/framework/plugins/imagelocality:go_default_library",
        "//pkg/scheduler/framework/plugins/interpodaffinity:go_default_library",
        "//pkg/scheduler/framework/plugins/nodeaffinity:go_default_library",
        "//pkg/scheduler/framework/plugins/nodename:go_default_library",
        "//pkg/scheduler/framework/plugins/nodeports:go_default_library",
        "//pkg/scheduler/framework/plugins/nodepreferavoidpods:go_default_library",
        "//pkg/scheduler/framework/plugins/noderesources:go_default_library",
        "//pkg/scheduler/framework/plugins/nodeunschedulable:go_default_library",
        "//pkg/scheduler/framework/plugins/nodevolumelimits:go_default_library",
        "//pkg/scheduler/framework/plugins/podtopologyspread:go_default_library",
        "//pkg/scheduler/framework/plugins/queuesort:go_default_library",
        "//pkg/scheduler/framework/plugins/tainttoleration:go_default_library",
        "//pkg/scheduler/framework/plugins/volumebinding:go_default_library",
        "//pkg/scheduler/framework/plugins/volumerestrictions:go_default_library",
        "//pkg/scheduler/framework/plugins/volumezone:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["registry_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/features:go_default_library",
        "//pkg/scheduler/apis/config:go_default_library",
        "//pkg/scheduler/framework/plugins/defaultbinder:go_default_library",
        "//pkg/scheduler/framework/plugins/defaultpodtopologyspread:go_default_library",
        "//pkg/scheduler/framework/plugins/imagelocality:go_default_library",
        "//pkg/scheduler/framework/plugins/interpodaffinity:go_default_library",
        "//pkg/scheduler/framework/plugins/nodeaffinity:go_default_library",
        "//pkg/scheduler/framework/plugins/nodename:go_default_library",
        "//pkg/scheduler/framework/plugins/nodeports:go_default_library",
        "//pkg/scheduler/framework/plugins/nodepreferavoidpods:go_default_library",
        "//pkg/scheduler/framework/plugins/noderesources:go_default_library",
        "//pkg/scheduler/framework/plugins/nodeunschedulable:go_default_library",
        "//pkg/scheduler/framework/plugins/nodevolumelimits:go_default_library",
        "//pkg/scheduler/framework/plugins/podtopologyspread:go_default_library",
        "//pkg/scheduler/framework/plugins/queuesort:go_default_library",
        "//pkg/scheduler/framework/plugins/tainttoleration:go_default_library",
        "//pkg/scheduler/framework/plugins/volumebinding:go_default_library",
        "//pkg/scheduler/framework/plugins/volumerestrictions:go_default_library",
        "//pkg/scheduler/framework/plugins/volumezone:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/component-base/featuregate/testing:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp:go_default_library",
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
)
