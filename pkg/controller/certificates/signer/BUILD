package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["cfssl_signer_test.go"],
    data = [
        "testdata/ca.crt",
        "testdata/ca.key",
        "testdata/kubelet.csr",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/api/certificates/v1beta1:go_default_library",
        "//staging/src/k8s.io/client-go/util/cert:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["cfssl_signer.go"],
    importpath = "k8s.io/kubernetes/pkg/controller/certificates/signer",
    deps = [
        "//pkg/controller/certificates:go_default_library",
        "//staging/src/k8s.io/api/certificates/v1beta1:go_default_library",
        "//staging/src/k8s.io/client-go/informers/certificates/v1beta1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/config:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/helpers:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/signer:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/signer/local:go_default_library",
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
