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
        "conntrack.go",
        "server.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-proxy/app/options:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/proxy:go_default_library",
        "//pkg/proxy/config:go_default_library",
        "//pkg/proxy/iptables:go_default_library",
        "//pkg/proxy/userspace:go_default_library",
        "//pkg/proxy/winuserspace:go_default_library",
        "//pkg/util/configz:go_default_library",
        "//pkg/util/dbus:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/iptables:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/util/netsh:go_default_library",
        "//pkg/util/node:go_default_library",
        "//pkg/util/oom:go_default_library",
        "//pkg/util/resourcecontainer:go_default_library",
        "//pkg/util/sysctl:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:github.com/spf13/cobra",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/client-go/kubernetes",
        "//vendor:k8s.io/client-go/kubernetes/typed/core/v1",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/tools/clientcmd",
        "//vendor:k8s.io/client-go/tools/clientcmd/api",
        "//vendor:k8s.io/client-go/tools/record",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["server_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-proxy/app/options:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/apis/componentconfig:go_default_library",
        "//pkg/util/iptables:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
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
        "//cmd/kube-proxy/app/options:all-srcs",
    ],
    tags = ["automanaged"],
)
