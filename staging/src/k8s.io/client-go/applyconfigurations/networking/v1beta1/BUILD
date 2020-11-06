load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "httpingresspath.go",
        "httpingressrulevalue.go",
        "ingress.go",
        "ingressbackend.go",
        "ingressclass.go",
        "ingressclassspec.go",
        "ingressrule.go",
        "ingressrulevalue.go",
        "ingressspec.go",
        "ingressstatus.go",
        "ingresstls.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/applyconfigurations/networking/v1beta1",
    importpath = "k8s.io/client-go/applyconfigurations/networking/v1beta1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/networking/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
        "//staging/src/k8s.io/client-go/applyconfigurations/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/applyconfigurations/meta/v1:go_default_library",
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
