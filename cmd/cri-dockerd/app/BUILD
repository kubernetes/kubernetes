load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["server.go"],
    importpath = "k8s.io/kubernetes/cmd/cri-dockerd/app",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/cri-dockerd/app/options:go_default_library",
        "//pkg/kubelet/dockershim:go_default_library",
        "//pkg/kubelet/dockershim/remote:go_default_library",
        "//pkg/kubelet/server/streaming:go_default_library",
        "//pkg/util/flag:go_default_library",
        "//pkg/version/verflag:go_default_library",
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//cmd/cri-dockerd/app/config:all-srcs",
        "//cmd/cri-dockerd/app/options:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
