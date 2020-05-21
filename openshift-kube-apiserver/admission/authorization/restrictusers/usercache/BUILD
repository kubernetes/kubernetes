load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["groups.go"],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/admission/authorization/restrictusers/usercache",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/github.com/openshift/api/user/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/user/informers/externalversions/user/v1:go_default_library",
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
