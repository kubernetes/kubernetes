package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
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
        "//vendor/github.com/containernetworking/cni/libcni:go_default_library",
        "//vendor/github.com/containernetworking/cni/pkg/types:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["cni_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/componentconfig:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/container/testing:go_default_library",
        "//pkg/kubelet/network:go_default_library",
        "//pkg/kubelet/network/cni/testing:go_default_library",
        "//pkg/kubelet/network/hostport:go_default_library",
        "//pkg/kubelet/network/testing:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//vendor/github.com/containernetworking/cni/pkg/types/020:go_default_library",
        "//vendor/github.com/stretchr/testify/mock:go_default_library",
        "//vendor/k8s.io/client-go/util/testing:go_default_library",
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
        "//pkg/kubelet/network/cni/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
