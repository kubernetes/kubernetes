package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["util_test.go"],
    importpath = "k8s.io/kubernetes/pkg/kubelet/util",
    library = ":go_default_library",
    deps = ["//vendor/github.com/stretchr/testify/assert:go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "util.go",
        "util_unsupported.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "util_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "util_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "util_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/kubelet/util",
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
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
        "//pkg/kubelet/util/queue:all-srcs",
        "//pkg/kubelet/util/sliceutils:all-srcs",
    ],
    tags = ["automanaged"],
)
