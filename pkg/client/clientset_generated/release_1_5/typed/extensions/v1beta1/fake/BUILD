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
        "doc.go",
        "fake_daemonset.go",
        "fake_deployment.go",
        "fake_deployment_expansion.go",
        "fake_extensions_client.go",
        "fake_ingress.go",
        "fake_job.go",
        "fake_podsecuritypolicy.go",
        "fake_replicaset.go",
        "fake_scale.go",
        "fake_scale_expansion.go",
        "fake_thirdpartyresource.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/release_1_5/typed/extensions/v1beta1:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/testing/core:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/watch:go_default_library",
    ],
)
