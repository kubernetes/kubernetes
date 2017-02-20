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
        "certificate_controller.go",
        "certificate_controller_utils.go",
        "cfssl_signer.go",
        "doc.go",
        "groupapprove.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/certificates/v1beta1:go_default_library",
        "//pkg/client/listers/certificates/v1beta1:go_default_library",
        "//pkg/controller:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = [
        "certificate_controller_test.go",
        "cfssl_signer_test.go",
        "groupapprove_test.go",
    ],
    data = [
        "testdata/ca.crt",
        "testdata/ca.key",
        "testdata/kubelet.csr",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/client/clientset_generated/clientset/fake:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions:go_default_library",
        "//pkg/controller:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
    ],
)
