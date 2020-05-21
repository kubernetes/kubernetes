load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "clusternetwork.go",
        "doc.go",
        "egressnetworkpolicy.go",
        "generated_expansion.go",
        "hostsubnet.go",
        "netnamespace.go",
        "network_client.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/client-go/network/clientset/versioned/typed/network/v1",
    importpath = "github.com/openshift/client-go/network/clientset/versioned/typed/network/v1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//vendor/github.com/openshift/api/network/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/network/clientset/versioned/scheme:go_default_library",
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
    visibility = ["//visibility:public"],
)
