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
        "configmap.go",
        "doc.go",
        "persistent_volume_claims.go",
        "pods.go",
        "registry.go",
        "replication_controllers.go",
        "resource_quotas.go",
        "secrets.go",
        "services.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "persistent_volume_claims_test.go",
        "pods_test.go",
        "services_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
