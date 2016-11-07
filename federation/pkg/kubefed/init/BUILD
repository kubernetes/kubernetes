package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = ["init.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/util:go_default_library",
        "//federation/pkg/kubefed/util:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/unversioned/clientcmd:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
        "//pkg/kubectl/cmd/templates:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/util/cert:go_default_library",
        "//pkg/util/cert/triple:go_default_library",
        "//pkg/util/intstr:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/version:go_default_library",
        "//vendor:github.com/spf13/cobra",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["init_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//federation/pkg/kubefed/testing:go_default_library",
        "//federation/pkg/kubefed/util:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/client/restclient/fake:go_default_library",
        "//pkg/client/typed/dynamic:go_default_library",
        "//pkg/client/unversioned/clientcmd:go_default_library",
        "//pkg/kubectl/cmd/testing:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/util/intstr:go_default_library",
        "//vendor:k8s.io/client-go/pkg/util/diff",
    ],
)
