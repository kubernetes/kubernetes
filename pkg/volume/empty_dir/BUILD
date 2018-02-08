package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "empty_dir.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "empty_dir_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "empty_dir_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "empty_dir_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "empty_dir_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "empty_dir_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "empty_dir_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "empty_dir_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "empty_dir_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "empty_dir_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "empty_dir_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "empty_dir_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/volume/empty_dir",
    deps = [
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/util/strings:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "empty_dir_test.go",
        ],
        "//conditions:default": [],
    }),
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/pkg/volume/empty_dir",
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/util/mount:go_default_library",
            "//pkg/volume:go_default_library",
            "//pkg/volume/testing:go_default_library",
            "//pkg/volume/util:go_default_library",
            "//vendor/k8s.io/api/core/v1:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
            "//vendor/k8s.io/client-go/util/testing:go_default_library",
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
)
