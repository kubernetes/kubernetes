load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "register.go",
        "stringsource.go",
        "types.go",
        "types_apiserver.go",
        "types_authentication.go",
        "types_build.go",
        "types_cluster_operator.go",
        "types_cluster_version.go",
        "types_console.go",
        "types_dns.go",
        "types_feature.go",
        "types_image.go",
        "types_infrastructure.go",
        "types_ingress.go",
        "types_network.go",
        "types_oauth.go",
        "types_operatorhub.go",
        "types_project.go",
        "types_proxy.go",
        "types_scheduling.go",
        "types_tlssecurityprofile.go",
        "zz_generated.deepcopy.go",
        "zz_generated.swagger_doc_generated.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/api/config/v1",
    importpath = "github.com/openshift/api/config/v1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
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
