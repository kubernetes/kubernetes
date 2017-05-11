package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "clientset.go",
        "doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//federation/client/clientset_generated/federation_internalclientset/typed/autoscaling/internalversion:go_default_library",
        "//federation/client/clientset_generated/federation_internalclientset/typed/batch/internalversion:go_default_library",
        "//federation/client/clientset_generated/federation_internalclientset/typed/core/internalversion:go_default_library",
        "//federation/client/clientset_generated/federation_internalclientset/typed/extensions/internalversion:go_default_library",
        "//federation/client/clientset_generated/federation_internalclientset/typed/federation/internalversion:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/client-go/discovery:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/util/flowcontrol:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//federation/client/clientset_generated/federation_internalclientset/fake:all-srcs",
        "//federation/client/clientset_generated/federation_internalclientset/scheme:all-srcs",
        "//federation/client/clientset_generated/federation_internalclientset/typed/autoscaling/internalversion:all-srcs",
        "//federation/client/clientset_generated/federation_internalclientset/typed/batch/internalversion:all-srcs",
        "//federation/client/clientset_generated/federation_internalclientset/typed/core/internalversion:all-srcs",
        "//federation/client/clientset_generated/federation_internalclientset/typed/extensions/internalversion:all-srcs",
        "//federation/client/clientset_generated/federation_internalclientset/typed/federation/internalversion:all-srcs",
    ],
    tags = ["automanaged"],
)
