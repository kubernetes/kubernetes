package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "util_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "util.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "util_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "util_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "util_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "util_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "util_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/kubelet/util",
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
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
        "//pkg/kubelet/util/cache:all-srcs",
        "//pkg/kubelet/util/format:all-srcs",
        "//pkg/kubelet/util/ioutils:all-srcs",
        "//pkg/kubelet/util/manager:all-srcs",
        "//pkg/kubelet/util/queue:all-srcs",
        "//pkg/kubelet/util/sliceutils:all-srcs",
        "//pkg/kubelet/util/store:all-srcs",
    ],
    tags = ["automanaged"],
)
