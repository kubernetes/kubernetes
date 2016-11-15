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
    srcs = ["replicaset.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/labels:go_default_library",
        "//pkg/util/pod:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
