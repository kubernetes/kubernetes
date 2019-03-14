package(default_visibility = ["//visibility:public"])

load(
    "//build:go.bzl",
    go_test = "go_test_conditional_pure",
)
load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_test(
    name = "go_default_test",
    srcs = [
        "e2e_kubeadm_suite_test.go",
        "kubeadm_test.go",
    ],
    out = "e2e_kubeadm.test",
    embed = [":go_default_library"],
    tags = ["e2e"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/rbac/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/cluster-bootstrap/token/api:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
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
    srcs = ["matchers.go"],
    importpath = "k8s.io/kubernetes/test/e2e_kubeadm",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
        "//vendor/github.com/onsi/gomega/gstruct:go_default_library",
    ],
)
