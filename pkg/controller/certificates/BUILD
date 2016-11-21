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
        "certificate_controller.go",
        "certificate_controller_utils.go",
        "doc.go",
        "groupapprove.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/certificates/v1alpha1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/release_1_5:go_default_library",
        "//pkg/client/clientset_generated/release_1_5/typed/certificates/v1alpha1:go_default_library",
        "//pkg/client/clientset_generated/release_1_5/typed/core/v1:go_default_library",
        "//pkg/client/record:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/cert:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/util/workqueue:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/cloudflare/cfssl/config",
        "//vendor:github.com/cloudflare/cfssl/signer",
        "//vendor:github.com/cloudflare/cfssl/signer/local",
        "//vendor:github.com/golang/glog",
    ],
)
