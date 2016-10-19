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
        "//federation/apis/federation/install:go_default_library",
        "//federation/client/clientset_generated/federation_release_1_5/typed/core/v1:go_default_library",
        "//federation/client/clientset_generated/federation_release_1_5/typed/extensions/v1beta1:go_default_library",
        "//federation/client/clientset_generated/federation_release_1_5/typed/federation/v1beta1:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/typed/discovery:go_default_library",
        "//pkg/util/flowcontrol:go_default_library",
        "//plugin/pkg/client/auth:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
