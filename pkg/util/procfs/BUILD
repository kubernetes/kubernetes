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
        "procfs.go",
        "procfs_fake.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "procfs_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "procfs_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "procfs_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/util/procfs",
    deps = select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "procfs_linux_test.go",
        ],
        "//conditions:default": [],
    }),
    data = [
        "example_proc_cgroup",
    ],
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/pkg/util/procfs",
    deps = select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
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
