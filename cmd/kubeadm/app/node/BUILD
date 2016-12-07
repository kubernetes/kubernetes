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
    srcs = [
        "bootstrap.go",
        "csr.go",
        "discovery.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/util:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/certificates:go_default_library",
        "//pkg/client/clientset_generated/release_1_5:go_default_library",
        "//pkg/client/clientset_generated/release_1_5/typed/certificates/v1alpha1:go_default_library",
        "//pkg/client/unversioned/clientcmd:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
        "//pkg/kubelet/util/csr:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/cert:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//vendor:github.com/square/go-jose",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "bootstrap_test.go",
        "csr_test.go",
        "discovery_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/client/clientset_generated/release_1_5:go_default_library",
        "//pkg/client/clientset_generated/release_1_5/typed/certificates/v1alpha1:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/typed/discovery:go_default_library",
        "//pkg/version:go_default_library",
    ],
)
