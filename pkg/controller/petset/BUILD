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
        "fakes.go",
        "identity_mappers.go",
        "iterator.go",
        "pet.go",
        "pet_set.go",
        "pet_set_utils.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "identity_mappers_test.go",
        "iterator_test.go",
        "pet_set_test.go",
        "pet_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
