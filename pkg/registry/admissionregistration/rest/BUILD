package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["storage_apiserver.go"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/admissionregistration:go_default_library",
        "//pkg/registry/admissionregistration/externaladmissionhookconfiguration/storage:go_default_library",
        "//pkg/registry/admissionregistration/initializerconfiguration/storage:go_default_library",
        "//vendor/k8s.io/api/admissionregistration/v1alpha1:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/registry/generic:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/registry/rest:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/server:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/server/storage:go_default_library",
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
)
