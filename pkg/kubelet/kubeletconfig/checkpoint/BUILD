package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "configmap_test.go",
        "download_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/kubelet/kubeletconfig/util/test:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/fake:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/github.com/davecgh/go-spew/spew:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "configmap.go",
        "download.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint",
    deps = [
        "//pkg/kubelet/apis/config:go_default_library",
        "//pkg/kubelet/apis/config/scheme:go_default_library",
        "//pkg/kubelet/kubeletconfig/status:go_default_library",
        "//pkg/kubelet/kubeletconfig/util/codec:go_default_library",
        "//pkg/kubelet/kubeletconfig/util/log:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/kubelet/config/v1beta1:go_default_library",
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
        "//pkg/kubelet/kubeletconfig/checkpoint/store:all-srcs",
    ],
    tags = ["automanaged"],
)
