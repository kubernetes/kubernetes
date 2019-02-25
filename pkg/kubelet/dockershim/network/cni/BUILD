package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "cni.go",
        "cni_others.go",
        "cni_windows.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/dockershim/network/cni",
    deps = [
        "//pkg/kubelet/apis/config:go_default_library",
        "//pkg/kubelet/apis/cri/runtime/v1alpha2:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/dockershim/network:go_default_library",
        "//pkg/util/bandwidth:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/github.com/containernetworking/cni/libcni:go_default_library",
        "//vendor/github.com/containernetworking/cni/pkg/types:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/containernetworking/cni/pkg/types/020:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = ["cni_test.go"],
    embed = [":go_default_library"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/kubelet/apis/config:go_default_library",
            "//pkg/kubelet/container:go_default_library",
            "//pkg/kubelet/container/testing:go_default_library",
            "//pkg/kubelet/dockershim/network:go_default_library",
            "//pkg/kubelet/dockershim/network/cni/testing:go_default_library",
            "//pkg/kubelet/dockershim/network/hostport:go_default_library",
            "//pkg/kubelet/dockershim/network/testing:go_default_library",
            "//staging/src/k8s.io/api/core/v1:go_default_library",
            "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
            "//staging/src/k8s.io/client-go/util/testing:go_default_library",
            "//vendor/github.com/containernetworking/cni/pkg/types/020:go_default_library",
            "//vendor/github.com/stretchr/testify/mock:go_default_library",
            "//vendor/k8s.io/utils/exec:go_default_library",
            "//vendor/k8s.io/utils/exec/testing:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//pkg/kubelet/dockershim/network/cni/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
