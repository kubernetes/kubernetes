package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "fsstore_test.go",
        "store_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/kubelet/apis/config:go_default_library",
        "//pkg/kubelet/apis/config/scheme:go_default_library",
        "//pkg/kubelet/kubeletconfig/checkpoint:go_default_library",
        "//pkg/kubelet/kubeletconfig/util/codec:go_default_library",
        "//pkg/kubelet/kubeletconfig/util/files:go_default_library",
        "//pkg/kubelet/kubeletconfig/util/test:go_default_library",
        "//pkg/util/filesystem:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/kubelet/config/v1beta1:go_default_library",
        "//vendor/github.com/davecgh/go-spew/spew:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "fakestore.go",
        "fsstore.go",
        "store.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint/store",
    deps = [
        "//pkg/kubelet/apis/config:go_default_library",
        "//pkg/kubelet/kubeletconfig/checkpoint:go_default_library",
        "//pkg/kubelet/kubeletconfig/configfiles:go_default_library",
        "//pkg/kubelet/kubeletconfig/util/files:go_default_library",
        "//pkg/kubelet/kubeletconfig/util/log:go_default_library",
        "//pkg/util/filesystem:go_default_library",
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
