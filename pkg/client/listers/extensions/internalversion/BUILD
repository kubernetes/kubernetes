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
        "daemonset.go",
        "deployment.go",
        "expansion_generated.go",
        "ingress.go",
        "networkpolicy.go",
        "podsecuritypolicy.go",
        "replicaset.go",
        "scale.go",
        "thirdpartyresource.go",
    ],
    tags = ["automanaged"],
)
