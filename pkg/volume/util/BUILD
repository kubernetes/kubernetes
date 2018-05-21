load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "atomic_writer.go",
        "attach_limit.go",
        "device_util.go",
        "doc.go",
        "error.go",
        "finalizer.go",
        "io_util.go",
        "metrics.go",
        "nested_volumes.go",
        "resize_util.go",
        "util.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "device_util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "device_util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "device_util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "device_util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "device_util_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "device_util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "device_util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "device_util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "device_util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "device_util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "device_util_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/volume/util",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/kubelet/apis:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/util/types:go_default_library",
        "//pkg/volume/util/volumepathhandler:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/api/storage/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/strategicpatch:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "nested_volumes_test.go",
        "resize_util_test.go",
        "util_test.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "atomic_writer_test.go",
            "device_util_linux_test.go",
        ],
        "//conditions:default": [],
    }),
    embed = [":go_default_library"],
    deps = [
        "//pkg/apis/core/install:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/util/slice:go_default_library",
        "//pkg/volume:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/client-go/util/testing:go_default_library",
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
        "//pkg/volume/util/fs:all-srcs",
        "//pkg/volume/util/nestedpendingoperations:all-srcs",
        "//pkg/volume/util/operationexecutor:all-srcs",
        "//pkg/volume/util/recyclerclient:all-srcs",
        "//pkg/volume/util/types:all-srcs",
        "//pkg/volume/util/volumepathhandler:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
