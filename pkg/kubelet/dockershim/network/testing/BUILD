package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "fake_host.go",
        "mock_network_plugin.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/dockershim/network/testing",
    deps = [
        "//pkg/kubelet/apis/config:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/dockershim/network:go_default_library",
        "//pkg/kubelet/dockershim/network/hostport:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/github.com/golang/mock/gomock:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["plugins_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/kubelet/apis/config:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/dockershim/network:go_default_library",
        "//pkg/util/sysctl/testing:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/github.com/golang/mock/gomock:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
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
