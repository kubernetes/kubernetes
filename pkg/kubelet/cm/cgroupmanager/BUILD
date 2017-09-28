load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "cgroup_manager_unsupported.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "cgroup_manager_linux.go",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//pkg/features:go_default_library",
            "//pkg/kubelet/cm/util:go_default_library",
            "//pkg/kubelet/metrics:go_default_library",
            "//vendor/github.com/docker/go-units:go_default_library",
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fs:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/systemd:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "cgroup_manager_linux_test.go",
            "cgroup_manager_test.go",
        ],
        "//conditions:default": [],
    }),
    library = ":go_default_library",
    deps = select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//pkg/kubelet/apis/kubeletconfig:go_default_library",
            "//pkg/kubelet/cm:go_default_library",
            "//vendor/k8s.io/api/core/v1:go_default_library",
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
