package(default_visibility = ["//visibility:public"])

load(
    "//build:go.bzl",
    go_test = "go_test_conditional_pure",
)
load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_test(
    name = "go_default_test",
    srcs = [
        "bootstrap_token_test.go",
        "cluster_info_test.go",
        "controlplane_nodes_test.go",
        "dns_addon_test.go",
        "e2e_kubeadm_suite_test.go",
        "kubeadm_certs_test.go",
        "kubeadm_config_test.go",
        "kubelet_config_test.go",
        "networking_test.go",
        "nodes_test.go",
        "proxy_addon_test.go",
    ],
    out = "e2e_kubeadm.test",
    embed = [":go_default_library"],
    tags = ["e2e"],
    deps = [
        "//staging/src/k8s.io/api/authorization/v1:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/rbac/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/version:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/cluster-bootstrap/token/api:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//test/e2e/framework/config:go_default_library",
        "//test/e2e/framework/skipper:go_default_library",
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/github.com/onsi/ginkgo/config:go_default_library",
        "//vendor/github.com/onsi/ginkgo/reporters:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/gopkg.in/yaml.v2:go_default_library",
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
        "//test/e2e_kubeadm/runner/local:all-srcs",
        "//test/e2e_kubeadm/tests:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "bootstrap_signer.go",
        "const.go",
        "framework.go",
        "util.go",
    ],
    importpath = "k8s.io/kubernetes/test/e2e_kubeadm",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/apps/v1:go_default_library",
        "//staging/src/k8s.io/api/authorization/v1:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/rbac/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
        "//vendor/github.com/onsi/gomega/gstruct:go_default_library",
    ],
)
