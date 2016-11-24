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
        "admission.go",
        "config.go",
        "doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/apis/imagepolicy/v1alpha1:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/util/cache:go_default_library",
        "//pkg/util/yaml:go_default_library",
        "//plugin/pkg/webhook:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "admission_test.go",
        "certs_test.go",
        "config_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/apis/imagepolicy/install:go_default_library",
        "//pkg/apis/imagepolicy/v1alpha1:go_default_library",
        "//pkg/auth/user:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/fake:go_default_library",
        "//pkg/client/unversioned/clientcmd/api/v1:go_default_library",
    ],
)
