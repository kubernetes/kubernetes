package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_test(
    name = "go_default_test",
    srcs = [
        "e2e_kubeadm_suite_test.go",
        "kubeadm_test.go",
    ],
    embed = [":go_default_library"],
    tags = ["e2e"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/rbac/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/tools/bootstrap/token/api:go_default_library",
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

genrule(
    name = "gen_e2e_kubeadm.test",
    testonly = 1,
    srcs = [":go_default_test"],
    outs = ["e2e_kubeadm.test"],
    cmd = "srcs=($(SRCS)); cp $$(dirname $${srcs[0]})/go_default_test $@;",
    output_to_bindir = 1,
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
