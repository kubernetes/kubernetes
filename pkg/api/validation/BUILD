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
        "events.go",
        "schema.go",
        "validation.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "events_test.go",
        "schema_test.go",
        "validation_test.go",
    ],
    data = [
        "testdata/v1/invalidPod.yaml",
        "testdata/v1/invalidPod1.json",
        "testdata/v1/invalidPod2.json",
        "testdata/v1/invalidPod3.json",
        "testdata/v1/invalidPod4.yaml",
        "testdata/v1/validPod.yaml",
        "//api/swagger-spec",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
