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
        "remote_image.go",
        "remote_runtime.go",
        "utils.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/cri/remote",
    deps = [
        "//pkg/kubelet/cri/remote/util:go_default_library",
        "//pkg/probe/exec:go_default_library",
        "//staging/src/k8s.io/component-base/logs/logreduction:go_default_library",
        "//staging/src/k8s.io/cri-api/pkg/apis:go_default_library",
        "//staging/src/k8s.io/cri-api/pkg/apis/runtime/v1alpha2:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
        "//vendor/google.golang.org/grpc/codes:go_default_library",
        "//vendor/google.golang.org/grpc/status:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
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
        "//pkg/kubelet/cri/remote/fake:all-srcs",
        "//pkg/kubelet/cri/remote/util:all-srcs",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = ["remote_runtime_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/kubelet/cri/remote/fake:go_default_library",
        "//staging/src/k8s.io/cri-api/pkg/apis:go_default_library",
        "//staging/src/k8s.io/cri-api/pkg/apis/testing:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
    ],
)
