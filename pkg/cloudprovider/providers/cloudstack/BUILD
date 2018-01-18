package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "cloudstack.go",
        "cloudstack_instances.go",
        "cloudstack_loadbalancer.go",
        "metadata.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "metadata_other.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "metadata_other.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "metadata_other.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "metadata_other.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "metadata_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "metadata_other.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "metadata_other.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "metadata_other.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "metadata_other.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "metadata_other.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "metadata_other.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/cloudprovider/providers/cloudstack",
    deps = [
        "//pkg/cloudprovider:go_default_library",
        "//pkg/controller:go_default_library",
        "//vendor/github.com/d2g/dhcp4:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/kardianos/osext:go_default_library",
        "//vendor/github.com/xanzy/go-cloudstack/cloudstack:go_default_library",
        "//vendor/gopkg.in/gcfg.v1:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/d2g/dhcp4client:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/github.com/d2g/dhcp4client:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/github.com/d2g/dhcp4client:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/github.com/d2g/dhcp4client:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/d2g/dhcp4client:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/github.com/d2g/dhcp4client:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/github.com/d2g/dhcp4client:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/github.com/d2g/dhcp4client:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/github.com/d2g/dhcp4client:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/github.com/d2g/dhcp4client:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/d2g/dhcp4client:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = ["cloudstack_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/pkg/cloudprovider/providers/cloudstack",
    deps = [
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
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
