load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "any_gen.go",
        "api_gen.go",
        "descriptor_gen.go",
        "doc.go",
        "duration_gen.go",
        "empty_gen.go",
        "field_mask_gen.go",
        "source_context_gen.go",
        "struct_gen.go",
        "timestamp_gen.go",
        "type_gen.go",
        "wrappers_gen.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/protobuf/internal/fieldnum",
    importpath = "google.golang.org/protobuf/internal/fieldnum",
    visibility = ["//vendor/google.golang.org/protobuf:__subpackages__"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
