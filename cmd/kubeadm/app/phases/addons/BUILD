load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["addons.go"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/phases/addons/dns:go_default_library",
        "//cmd/kubeadm/app/phases/addons/proxy:go_default_library",
        "//cmd/kubeadm/app/util/apiclient:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
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
        "//cmd/kubeadm/app/phases/addons/dns:all-srcs",
        "//cmd/kubeadm/app/phases/addons/proxy:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
