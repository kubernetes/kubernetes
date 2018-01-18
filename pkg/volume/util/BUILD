package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "atomic_writer.go",
        "device_util.go",
        "doc.go",
        "error.go",
        "finalizer.go",
        "io_util.go",
        "metrics.go",
        "util.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "device_util_unsupported.go",
            "fs_unsupported.go",
            "util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "device_util_unsupported.go",
            "fs.go",
            "util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "device_util_unsupported.go",
            "fs_unsupported.go",
            "util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "device_util_unsupported.go",
            "fs_unsupported.go",
            "util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "device_util_linux.go",
            "fs.go",
            "util_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "device_util_unsupported.go",
            "fs_unsupported.go",
            "util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "device_util_unsupported.go",
            "fs_unsupported.go",
            "util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "device_util_unsupported.go",
            "fs_unsupported.go",
            "util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "device_util_unsupported.go",
            "fs_unsupported.go",
            "util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "device_util_unsupported.go",
            "fs_unsupported.go",
            "util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "device_util_unsupported.go",
            "fs_unsupported.go",
            "util_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/volume/util",
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/kubelet/apis:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/api/storage/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = [
        "util_test.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "atomic_writer_test.go",
            "device_util_linux_test.go",
        ],
        "//conditions:default": [],
    }),
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/pkg/volume/util",
    deps = [
        "//pkg/apis/core/install:go_default_library",
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
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
        "//pkg/volume/util/nestedpendingoperations:all-srcs",
        "//pkg/volume/util/operationexecutor:all-srcs",
        "//pkg/volume/util/types:all-srcs",
        "//pkg/volume/util/volumehelper:all-srcs",
    ],
    tags = ["automanaged"],
)
