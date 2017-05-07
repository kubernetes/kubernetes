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
        "import_known_versions.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//federation/apis/federation/install:go_default_library",
        "//federation/client/clientset_generated/federation_clientset/typed/autoscaling/v1:go_default_library",
        "//federation/client/clientset_generated/federation_clientset/typed/batch/v1:go_default_library",
        "//federation/client/clientset_generated/federation_clientset/typed/core/v1:go_default_library",
        "//federation/client/clientset_generated/federation_clientset/typed/extensions/v1beta1:go_default_library",
        "//federation/client/clientset_generated/federation_clientset/typed/federation/v1beta1:go_default_library",
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
        "//federation/client/clientset_generated/federation_clientset/fake:all-srcs",
        "//federation/client/clientset_generated/federation_clientset/scheme:all-srcs",
        "//federation/client/clientset_generated/federation_clientset/typed/autoscaling/v1:all-srcs",
        "//federation/client/clientset_generated/federation_clientset/typed/batch/v1:all-srcs",
        "//federation/client/clientset_generated/federation_clientset/typed/core/v1:all-srcs",
        "//federation/client/clientset_generated/federation_clientset/typed/extensions/v1beta1:all-srcs",
        "//federation/client/clientset_generated/federation_clientset/typed/federation/v1beta1:all-srcs",
    ],
    tags = ["automanaged"],
)
