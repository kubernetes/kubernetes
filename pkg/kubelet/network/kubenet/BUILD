package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "kubenet.go",
        "kubenet_linux.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/componentconfig:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/network:go_default_library",
        "//pkg/kubelet/network/hostport:go_default_library",
        "//pkg/util/bandwidth:go_default_library",
        "//pkg/util/dbus:go_default_library",
        "//pkg/util/ebtables:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/iptables:go_default_library",
        "//pkg/util/sysctl:go_default_library",
        "//vendor:github.com/containernetworking/cni/libcni",
        "//vendor:github.com/containernetworking/cni/pkg/types",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/vishvananda/netlink",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["kubenet_linux_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/componentconfig:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/network:go_default_library",
        "//pkg/kubelet/network/cni/testing:go_default_library",
        "//pkg/kubelet/network/hostport/testing:go_default_library",
        "//pkg/kubelet/network/testing:go_default_library",
        "//pkg/util/bandwidth:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/iptables/testing:go_default_library",
        "//pkg/util/sysctl/testing:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:github.com/stretchr/testify/mock",
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
