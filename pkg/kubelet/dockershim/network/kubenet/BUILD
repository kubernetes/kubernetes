package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "kubenet.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "kubenet_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "kubenet_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "kubenet_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "kubenet_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "kubenet_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "kubenet_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "kubenet_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "kubenet_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "kubenet_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "kubenet_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "kubenet_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/kubelet/dockershim/network/kubenet",
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//pkg/kubelet/apis/kubeletconfig:go_default_library",
            "//pkg/kubelet/container:go_default_library",
            "//pkg/kubelet/dockershim/network:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//pkg/kubelet/apis/kubeletconfig:go_default_library",
            "//pkg/kubelet/container:go_default_library",
            "//pkg/kubelet/dockershim/network:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//pkg/kubelet/apis/kubeletconfig:go_default_library",
            "//pkg/kubelet/container:go_default_library",
            "//pkg/kubelet/dockershim/network:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//pkg/kubelet/apis/kubeletconfig:go_default_library",
            "//pkg/kubelet/container:go_default_library",
            "//pkg/kubelet/dockershim/network:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/kubelet/apis/kubeletconfig:go_default_library",
            "//pkg/kubelet/container:go_default_library",
            "//pkg/kubelet/dockershim/network:go_default_library",
            "//pkg/kubelet/dockershim/network/hostport:go_default_library",
            "//pkg/util/bandwidth:go_default_library",
            "//pkg/util/dbus:go_default_library",
            "//pkg/util/ebtables:go_default_library",
            "//pkg/util/iptables:go_default_library",
            "//pkg/util/sysctl:go_default_library",
            "//vendor/github.com/containernetworking/cni/libcni:go_default_library",
            "//vendor/github.com/containernetworking/cni/pkg/types:go_default_library",
            "//vendor/github.com/containernetworking/cni/pkg/types/020:go_default_library",
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/github.com/vishvananda/netlink:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/net:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//pkg/kubelet/apis/kubeletconfig:go_default_library",
            "//pkg/kubelet/container:go_default_library",
            "//pkg/kubelet/dockershim/network:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//pkg/kubelet/apis/kubeletconfig:go_default_library",
            "//pkg/kubelet/container:go_default_library",
            "//pkg/kubelet/dockershim/network:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//pkg/kubelet/apis/kubeletconfig:go_default_library",
            "//pkg/kubelet/container:go_default_library",
            "//pkg/kubelet/dockershim/network:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//pkg/kubelet/apis/kubeletconfig:go_default_library",
            "//pkg/kubelet/container:go_default_library",
            "//pkg/kubelet/dockershim/network:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//pkg/kubelet/apis/kubeletconfig:go_default_library",
            "//pkg/kubelet/container:go_default_library",
            "//pkg/kubelet/dockershim/network:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//pkg/kubelet/apis/kubeletconfig:go_default_library",
            "//pkg/kubelet/container:go_default_library",
            "//pkg/kubelet/dockershim/network:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "kubenet_linux_test.go",
        ],
        "//conditions:default": [],
    }),
    embed = [":go_default_library"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/kubelet/apis/kubeletconfig:go_default_library",
            "//pkg/kubelet/container:go_default_library",
            "//pkg/kubelet/dockershim/network:go_default_library",
            "//pkg/kubelet/dockershim/network/cni/testing:go_default_library",
            "//pkg/kubelet/dockershim/network/hostport/testing:go_default_library",
            "//pkg/kubelet/dockershim/network/testing:go_default_library",
            "//pkg/util/bandwidth:go_default_library",
            "//pkg/util/iptables/testing:go_default_library",
            "//pkg/util/sysctl/testing:go_default_library",
            "//vendor/github.com/stretchr/testify/assert:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
