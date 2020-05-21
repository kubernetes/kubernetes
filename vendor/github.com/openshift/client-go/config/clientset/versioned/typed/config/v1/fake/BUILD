load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fake_apiserver.go",
        "fake_authentication.go",
        "fake_build.go",
        "fake_clusteroperator.go",
        "fake_clusterversion.go",
        "fake_config_client.go",
        "fake_console.go",
        "fake_dns.go",
        "fake_featuregate.go",
        "fake_image.go",
        "fake_infrastructure.go",
        "fake_ingress.go",
        "fake_network.go",
        "fake_oauth.go",
        "fake_operatorhub.go",
        "fake_project.go",
        "fake_proxy.go",
        "fake_scheduler.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/client-go/config/clientset/versioned/typed/config/v1/fake",
    importpath = "github.com/openshift/client-go/config/clientset/versioned/typed/config/v1/fake",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/testing:go_default_library",
        "//vendor/github.com/openshift/api/config/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/config/clientset/versioned/typed/config/v1:go_default_library",
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
