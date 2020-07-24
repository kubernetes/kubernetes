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
    importpath = "k8s.io/kubernetes/pkg/kubelet/dockershim/cm",
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//pkg/kubelet/cm:go_default_library",
            "//pkg/kubelet/dockershim/libdocker:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/util/version:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fs:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//pkg/kubelet/dockershim/libdocker:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//pkg/kubelet/dockershim/libdocker:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//pkg/kubelet/dockershim/libdocker:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:ios": [
            "//pkg/kubelet/dockershim/libdocker:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/kubelet/cm:go_default_library",
            "//pkg/kubelet/dockershim/libdocker:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/util/version:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fs:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//pkg/kubelet/dockershim/libdocker:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//pkg/kubelet/dockershim/libdocker:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//pkg/kubelet/dockershim/libdocker:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//pkg/kubelet/dockershim/libdocker:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//pkg/kubelet/dockershim/libdocker:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//pkg/kubelet/dockershim/libdocker:go_default_library",
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
