load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "daemonset.go",
        "deployment.go",
        "ingress.go",
        "networkpolicy.go",
        "podsecuritypolicy.go",
        "replicaset.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/listers/extensions/v1beta1_fakelister",
    importpath = "k8s.io/client-go/listers/extensions/v1beta1_fakelister",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/extensions/v1beta1:go_default_library",
        "//staging/src/k8s.io/client-go/listers/extensions/v1beta1:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
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
