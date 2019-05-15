package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "container_manager.go",
        "container_manager_linux.go",
        "container_manager_unsupported.go",
        "container_manager_windows.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubelet/pkg/dockershim/cm",
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//cmd/kubelet/pkg/dockershim/libdocker:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//cmd/kubelet/pkg/dockershim/libdocker:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//cmd/kubelet/pkg/dockershim/libdocker:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//cmd/kubelet/pkg/dockershim/libdocker:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//cmd/kubelet/pkg/cm:go_default_library",
            "//cmd/kubelet/pkg/dockershim/libdocker:go_default_library",
            "//cmd/kubelet/pkg/qos:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/util/version:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fs:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//cmd/kubelet/pkg/dockershim/libdocker:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//cmd/kubelet/pkg/dockershim/libdocker:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//cmd/kubelet/pkg/dockershim/libdocker:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//cmd/kubelet/pkg/dockershim/libdocker:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//cmd/kubelet/pkg/dockershim/libdocker:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//cmd/kubelet/pkg/dockershim/libdocker:go_default_library",
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
