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
        "balanced_resource_allocation.go",
        "image_locality.go",
        "interpod_affinity.go",
        "least_requested.go",
        "metadata.go",
        "most_requested.go",
        "node_affinity.go",
        "node_label.go",
        "node_prefer_avoid_pods.go",
        "selector_spreading.go",
        "taint_toleration.go",
        "test_util.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "balanced_resource_allocation_test.go",
        "image_locality_test.go",
        "interpod_affinity_test.go",
        "least_requested_test.go",
        "most_requested_test.go",
        "node_affinity_test.go",
        "node_label_test.go",
        "node_prefer_avoid_pods_test.go",
        "priorities_test.go",
        "selector_spreading_test.go",
        "taint_toleration_test.go",
    ],
    library = "go_default_library",
    tags = [
        "automanaged",
        "skip",
    ],
    deps = [],
)
