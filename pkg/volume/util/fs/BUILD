load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "fs.go",
        "fs_unsupported.go",
        "fs_windows.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/volume/util/fs",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
            "//vendor/golang.org/x/sys/windows:go_default_library",
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
    visibility = ["//visibility:public"],
)
