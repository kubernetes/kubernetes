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
        "cmd.go",
        "completion.go",
        "defaults.go",
        "init.go",
        "join.go",
        "reset.go",
        "token.go",
        "version.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/v1alpha1:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/validation:go_default_library",
        "//cmd/kubeadm/app/cmd/phases:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/discovery:go_default_library",
        "//cmd/kubeadm/app/master:go_default_library",
        "//cmd/kubeadm/app/node:go_default_library",
        "//cmd/kubeadm/app/phases/addons:go_default_library",
        "//cmd/kubeadm/app/phases/apiconfig:go_default_library",
        "//cmd/kubeadm/app/phases/certs:go_default_library",
        "//cmd/kubeadm/app/phases/kubeconfig:go_default_library",
        "//cmd/kubeadm/app/phases/token:go_default_library",
        "//cmd/kubeadm/app/preflight:go_default_library",
        "//cmd/kubeadm/app/util:go_default_library",
        "//cmd/kubeadm/app/util/kubeconfig:go_default_library",
        "//cmd/kubeadm/app/util/token:go_default_library",
        "//pkg/bootstrap/api:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/printers:go_default_library",
        "//pkg/util/i18n:go_default_library",
        "//pkg/util/initsystem:go_default_library",
        "//pkg/version:go_default_library",
        "//vendor:github.com/blang/semver",
        "//vendor:github.com/renstrom/dedent",
        "//vendor:github.com/spf13/cobra",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apiserver/pkg/util/flag",
        "//vendor:k8s.io/client-go/kubernetes",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/util/cert",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "reset_test.go",
        "token_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/preflight:go_default_library",
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
        "//cmd/kubeadm/app/cmd/phases:all-srcs",
    ],
    tags = ["automanaged"],
)
