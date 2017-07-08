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
        "certs.go",
        "kubeconfig.go",
        "phase.go",
        "preflight.go",
        "selfhosting.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/v1alpha1:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/validation:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/phases/certs:go_default_library",
        "//cmd/kubeadm/app/phases/certs/pkiutil:go_default_library",
        "//cmd/kubeadm/app/phases/kubeconfig:go_default_library",
        "//cmd/kubeadm/app/phases/selfhosting:go_default_library",
        "//cmd/kubeadm/app/preflight:go_default_library",
        "//cmd/kubeadm/app/util:go_default_library",
        "//cmd/kubeadm/app/util/config:go_default_library",
        "//cmd/kubeadm/app/util/kubeconfig:go_default_library",
        "//pkg/api:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["certs_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm/install:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/phases/certs/pkiutil:go_default_library",
        "//vendor/github.com/renstrom/dedent:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
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
