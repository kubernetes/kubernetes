load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "hugepage_handler_linux.go",
        "hugepage_handler_unsupported.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/cm/hugepagehandler",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/cri-api/pkg/apis/runtime/v1alpha2:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/apis/core/v1/helper:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
            "//vendor/github.com/docker/go-units:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fs:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["hugepage_handler_linux_test.go"],
    embed = [":go_default_library"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/apis/core:go_default_library",
            "//staging/src/k8s.io/api/core/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
            "//staging/src/k8s.io/cri-api/pkg/apis/runtime/v1alpha2:go_default_library",
        ],
        "//conditions:default": [],
    }),
)
