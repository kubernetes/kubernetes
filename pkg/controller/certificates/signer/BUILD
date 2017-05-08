package(default_visibility = ["//visibility:public"])

licenses(["notice"])

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
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/certificates/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/util/cert:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["cfssl_signer.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/certificates/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/certificates/v1beta1:go_default_library",
        "//pkg/controller/certificates:go_default_library",
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
