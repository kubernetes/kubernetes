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
        "config.go",
        "default_storage_factory_builder.go",
        "discovery.go",
        "doc.go",
        "genericapiserver.go",
        "healthz.go",
        "hooks.go",
        "resource_config.go",
        "resource_encoding_config.go",
        "reststorage_interfaces.go",
        "serve.go",
        "services.go",
        "storage_factory.go",
        "tunneler.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "default_storage_factory_builder_test.go",
        "genericapiserver_test.go",
        "resource_config_test.go",
        "serve_test.go",
        "server_run_options_test.go",
        "storage_factory_test.go",
        "tunneler_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
