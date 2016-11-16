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
    srcs = ["kubenet_linux.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/componentconfig:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/network:go_default_library",
        "//pkg/kubelet/network/hostport:go_default_library",
        "//pkg/util/bandwidth:go_default_library",
        "//pkg/util/dbus:go_default_library",
        "//pkg/util/ebtables:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/iptables:go_default_library",
        "//pkg/util/net:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/sysctl:go_default_library",
        "//vendor:github.com/containernetworking/cni/libcni",
        "//vendor:github.com/containernetworking/cni/pkg/types",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/vishvananda/netlink",
        "//vendor:github.com/vishvananda/netlink/nl",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["kubenet_linux_test.go"],
    library = "go_default_library",
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
