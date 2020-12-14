load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "alpha.go",
        "certs.go",
        "kubeconfig.go",
        "selfhosting.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/cmd/alpha",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/scheme:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/v1beta2:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/validation:go_default_library",
        "//cmd/kubeadm/app/cmd/options:go_default_library",
        "//cmd/kubeadm/app/cmd/phases:go_default_library",
        "//cmd/kubeadm/app/cmd/util:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/features:go_default_library",
        "//cmd/kubeadm/app/phases/certs:go_default_library",
        "//cmd/kubeadm/app/phases/certs/renewal:go_default_library",
        "//cmd/kubeadm/app/phases/copycerts:go_default_library",
        "//cmd/kubeadm/app/phases/kubeconfig:go_default_library",
        "//cmd/kubeadm/app/phases/selfhosting:go_default_library",
        "//cmd/kubeadm/app/util/apiclient:go_default_library",
        "//cmd/kubeadm/app/util/config:go_default_library",
        "//cmd/kubeadm/app/util/kubeconfig:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/duration:go_default_library",
        "//vendor/github.com/lithammer/dedent:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "certs_test.go",
        "kubeconfig_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/v1beta2:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/phases/certs:go_default_library",
        "//cmd/kubeadm/app/phases/kubeconfig:go_default_library",
        "//cmd/kubeadm/app/util/pkiutil:go_default_library",
        "//cmd/kubeadm/test:go_default_library",
        "//cmd/kubeadm/test/cmd:go_default_library",
        "//cmd/kubeadm/test/kubeconfig:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/client-go/tools/clientcmd:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
        "//vendor/sigs.k8s.io/yaml:go_default_library",
    ],
)
