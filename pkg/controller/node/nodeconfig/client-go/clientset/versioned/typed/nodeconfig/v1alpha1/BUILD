load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "generated_expansion.go",
        "nodeconfig_client.go",
        "nodeconfigsourcepool.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/controller/node/nodeconfig/client-go/clientset/versioned/typed/nodeconfig/v1alpha1",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/controller/node/nodeconfig/client-go/clientset/versioned/scheme:go_default_library",
        "//pkg/controller/node/nodeconfig/v1alpha1:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
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
        "//pkg/controller/node/nodeconfig/client-go/clientset/versioned/typed/nodeconfig/v1alpha1/fake:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
