load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "apiserver.go",
        "authentication.go",
        "build.go",
        "clusteroperator.go",
        "clusterversion.go",
        "config_client.go",
        "console.go",
        "dns.go",
        "doc.go",
        "featuregate.go",
        "generated_expansion.go",
        "image.go",
        "infrastructure.go",
        "ingress.go",
        "network.go",
        "oauth.go",
        "operatorhub.go",
        "project.go",
        "proxy.go",
        "scheduler.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/client-go/config/clientset/versioned/typed/config/v1",
    importpath = "github.com/openshift/client-go/config/clientset/versioned/typed/config/v1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//vendor/github.com/openshift/api/config/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/config/clientset/versioned/scheme:go_default_library",
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
        "//vendor/github.com/openshift/client-go/config/clientset/versioned/typed/config/v1/fake:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
