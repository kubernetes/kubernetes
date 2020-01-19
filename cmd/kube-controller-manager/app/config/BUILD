load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["config.go"],
    importpath = "k8s.io/kubernetes/cmd/kube-controller-manager/app/config",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/controller/apis/config:go_default_library",
        "//pkg/controller/apis/config/scheme:go_default_library",
        "//pkg/util/filesystem:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/tools/record:go_default_library",
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
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["config_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/controller/apis/config:go_default_library",
        "//pkg/controller/apis/config/scheme:go_default_library",
        "//pkg/kubelet/kubeletconfig/util/files:go_default_library",
        "//pkg/util/filesystem:go_default_library",
        "//staging/src/k8s.io/kube-controller-manager/config/v1alpha1:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
    ],
)
