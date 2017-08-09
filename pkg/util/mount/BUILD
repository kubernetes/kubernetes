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
        "fake.go",
        "mount.go",
        "mount_unsupported.go",
        "nsenter_mount_unsupported.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "mount_linux.go",
            "nsenter_mount.go",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
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
            "mount_linux_test.go",
            "nsenter_mount_test.go",
        ],
        "//conditions:default": [],
    }),
    library = ":go_default_library",
    deps = [
        "//vendor/k8s.io/utils/exec:go_default_library",
        "//vendor/k8s.io/utils/exec/testing:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
