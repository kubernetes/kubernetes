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
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/certificates/v1alpha1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/certificates/v1alpha1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/core/v1:go_default_library",
        "//pkg/client/record:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/util/workqueue:go_default_library",
        "//vendor:github.com/cloudflare/cfssl/config",
        "//vendor:github.com/cloudflare/cfssl/helpers",
        "//vendor:github.com/cloudflare/cfssl/signer",
        "//vendor:github.com/cloudflare/cfssl/signer/local",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apimachinery/pkg/watch",
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
        "//pkg/apis/certificates/v1alpha1:go_default_library",
        "//pkg/util/cert:go_default_library",
    ],
)
