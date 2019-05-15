package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["configfiles.go"],
    importpath = "k8s.io/kubernetes/cmd/kubelet/pkg/kubeletconfig/configfiles",
    deps = [
        "//cmd/kubelet/pkg/apis/config:go_default_library",
        "//cmd/kubelet/pkg/apis/config/scheme:go_default_library",
        "//cmd/kubelet/pkg/kubeletconfig/util/codec:go_default_library",
        "//pkg/util/filesystem:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = ["configfiles_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//cmd/kubelet/pkg/apis/config:go_default_library",
        "//cmd/kubelet/pkg/apis/config/scheme:go_default_library",
        "//cmd/kubelet/pkg/kubeletconfig/util/files:go_default_library",
        "//cmd/kubelet/pkg/kubeletconfig/util/test:go_default_library",
        "//pkg/util/filesystem:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/kubelet/config/v1beta1:go_default_library",
    ],
)
