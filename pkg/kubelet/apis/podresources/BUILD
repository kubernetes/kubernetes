load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "constants.go",
        "server.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/apis/podresources",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/apis/podresources/v1alpha1:go_default_library",
        "//pkg/kubelet/util:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["server_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/kubelet/apis/podresources/v1alpha1:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/github.com/stretchr/testify/mock:go_default_library",
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
        "//pkg/kubelet/apis/podresources/v1alpha1:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
