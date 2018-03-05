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
        "exec.go",
        "exec_mount_unsupported.go",
        "fake.go",
        "mount.go",
        "mount_unsupported.go",
        "nsenter_mount_unsupported.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "exec_mount.go",
            "mount_linux.go",
            "nsenter_mount.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "mount_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/util/mount",
    deps = [
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//pkg/util/io:go_default_library",
            "//pkg/util/nsenter:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = [
        "safe_format_and_mount_test.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "exec_mount_test.go",
            "mount_linux_test.go",
            "nsenter_mount_test.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "mount_windows_test.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/util/mount",
    library = ":go_default_library",
    deps = [
        "//vendor/k8s.io/utils/exec/testing:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//vendor/github.com/golang/glog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "//vendor/github.com/stretchr/testify/assert:go_default_library",
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
