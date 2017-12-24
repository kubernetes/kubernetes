package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "addons.go",
        "bootstraptoken.go",
        "certs.go",
        "controlplane.go",
        "etcd.go",
        "kubeconfig.go",
        "markmaster.go",
        "phase.go",
        "preflight.go",
        "selfhosting.go",
        "uploadconfig.go",
        "util.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases",
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/v1alpha1:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/validation:go_default_library",
        "//cmd/kubeadm/app/cmd/util:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/features:go_default_library",
        "//cmd/kubeadm/app/phases/addons/dns:go_default_library",
        "//cmd/kubeadm/app/phases/addons/proxy:go_default_library",
        "//cmd/kubeadm/app/phases/bootstraptoken/clusterinfo:go_default_library",
        "//cmd/kubeadm/app/phases/bootstraptoken/node:go_default_library",
        "//cmd/kubeadm/app/phases/certs:go_default_library",
        "//cmd/kubeadm/app/phases/certs/pkiutil:go_default_library",
        "//cmd/kubeadm/app/phases/controlplane:go_default_library",
        "//cmd/kubeadm/app/phases/etcd:go_default_library",
        "//cmd/kubeadm/app/phases/kubeconfig:go_default_library",
        "//cmd/kubeadm/app/phases/markmaster:go_default_library",
        "//cmd/kubeadm/app/phases/selfhosting:go_default_library",
        "//cmd/kubeadm/app/phases/uploadconfig:go_default_library",
        "//cmd/kubeadm/app/preflight:go_default_library",
        "//cmd/kubeadm/app/util:go_default_library",
        "//cmd/kubeadm/app/util/apiclient:go_default_library",
        "//cmd/kubeadm/app/util/config:go_default_library",
        "//cmd/kubeadm/app/util/kubeconfig:go_default_library",
        "//cmd/kubeadm/app/util/pubkeypin:go_default_library",
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/bootstrap/api:go_default_library",
        "//pkg/util/normalizer:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "addons_test.go",
        "certs_test.go",
        "controlplane_test.go",
        "etcd_test.go",
        "kubeconfig_test.go",
    ],
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases",
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/install:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/phases/certs/pkiutil:go_default_library",
        "//cmd/kubeadm/test:go_default_library",
        "//cmd/kubeadm/test/cmd:go_default_library",
        "//cmd/kubeadm/test/kubeconfig:go_default_library",
        "//pkg/util/node:go_default_library",
        "//vendor/k8s.io/client-go/tools/clientcmd:go_default_library",
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
