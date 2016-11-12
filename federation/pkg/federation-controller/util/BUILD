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
        "backoff.go",
        "cluster_util.go",
        "configmap.go",
        "delaying_deliverer.go",
        "federated_informer.go",
        "federated_updater.go",
        "handlers.go",
        "meta.go",
        "secret.go",
        "versionize_listoptions.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "delaying_deliverer_test.go",
        "federated_informer_test.go",
        "federated_updater_test.go",
        "handlers_test.go",
        "meta_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
