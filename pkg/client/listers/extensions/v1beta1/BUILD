package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "zz_generated.daemonset.go",
        "zz_generated.deployment.go",
        "zz_generated.ingress.go",
        "zz_generated.job.go",
        "zz_generated.podsecuritypolicy.go",
        "zz_generated.replicaset.go",
        "zz_generated.scale.go",
        "zz_generated.thirdpartyresource.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/errors:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/labels:go_default_library",
    ],
)
