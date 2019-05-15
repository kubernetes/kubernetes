package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "util_unix_test.go",
        "util_windows_test.go",
    ],
    embed = [":go_default_library"],
    deps = select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/github.com/stretchr/testify/assert:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/github.com/stretchr/testify/assert:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/stretchr/testify/assert:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/stretchr/testify/assert:go_default_library",
            "//vendor/github.com/stretchr/testify/require:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_library(
    name = "go_default_library",
    srcs = [
        "boottime_util_darwin.go",
        "boottime_util_linux.go",
        "doc.go",
        "util.go",
        "util_unix.go",
        "util_unsupported.go",
        "util_windows.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubelet/pkg/util",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/Microsoft/go-winio:go_default_library",
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
        "//cmd/kubelet/pkg/util/cache:all-srcs",
        "//cmd/kubelet/pkg/util/format:all-srcs",
        "//cmd/kubelet/pkg/util/ioutils:all-srcs",
        "//cmd/kubelet/pkg/util/logreduction:all-srcs",
        "//cmd/kubelet/pkg/util/manager:all-srcs",
        "//cmd/kubelet/pkg/util/oom:all-srcs",
        "//cmd/kubelet/pkg/util/pluginwatcher:all-srcs",
        "//cmd/kubelet/pkg/util/queue:all-srcs",
        "//cmd/kubelet/pkg/util/sliceutils:all-srcs",
        "//cmd/kubelet/pkg/util/store:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
