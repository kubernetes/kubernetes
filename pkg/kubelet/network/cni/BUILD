package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = ["cni.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/componentconfig:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/network:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//vendor:github.com/containernetworking/cni/libcni",
        "//vendor:github.com/containernetworking/cni/pkg/types",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["cni_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/componentconfig:go_default_library",
        "//pkg/client/clientset_generated/release_1_5:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/container/testing:go_default_library",
        "//pkg/kubelet/network:go_default_library",
        "//pkg/kubelet/network/cni/testing:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/testing:go_default_library",
        "//vendor:github.com/containernetworking/cni/pkg/types",
        "//vendor:github.com/stretchr/testify/mock",
    ],
)
