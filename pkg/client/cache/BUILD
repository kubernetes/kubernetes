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
        "controller.go",
        "delta_fifo.go",
        "doc.go",
        "expiration_cache.go",
        "expiration_cache_fakes.go",
        "fake_custom_store.go",
        "fifo.go",
        "index.go",
        "listers.go",
        "listers_core.go",
        "listers_extensions.go",
        "listers_rbac.go",
        "listwatch.go",
        "mutation_detector.go",
        "reflector.go",
        "shared_informer.go",
        "store.go",
        "thread_safe_store.go",
        "undelta_store.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "controller_test.go",
        "delta_fifo_test.go",
        "expiration_cache_test.go",
        "fifo_test.go",
        "index_test.go",
        "listers_test.go",
        "listwatch_test.go",
        "mutation_detector_test.go",
        "processor_listener_test.go",
        "reflector_test.go",
        "store_test.go",
        "undelta_store_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
