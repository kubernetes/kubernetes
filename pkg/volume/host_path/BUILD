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
        "host_path.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/uuid:go_default_library",
        "//pkg/volume:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["host_path_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/resource:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/clientset_generated/release_1_5/fake:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util:go_default_library",
        "//pkg/util/uuid:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/testing:go_default_library",
    ],
)
