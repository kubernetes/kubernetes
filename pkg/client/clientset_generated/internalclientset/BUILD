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
        "clientset.go",
        "doc.go",
        "import_known_versions.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/install:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/apis/apps/install:go_default_library",
        "//pkg/apis/authentication/install:go_default_library",
        "//pkg/apis/authorization/install:go_default_library",
        "//pkg/apis/autoscaling/install:go_default_library",
        "//pkg/apis/batch/install:go_default_library",
        "//pkg/apis/certificates/install:go_default_library",
        "//pkg/apis/componentconfig/install:go_default_library",
        "//pkg/apis/extensions/install:go_default_library",
        "//pkg/apis/policy/install:go_default_library",
        "//pkg/apis/rbac/install:go_default_library",
        "//pkg/apis/storage/install:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/apps/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/authentication/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/authorization/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/autoscaling/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/batch/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/certificates/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/core/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/policy/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/storage/internalversion:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/typed/discovery:go_default_library",
        "//pkg/util/flowcontrol:go_default_library",
        "//plugin/pkg/client/auth:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
