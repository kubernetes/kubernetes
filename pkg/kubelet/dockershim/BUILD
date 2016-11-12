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
        "convert.go",
        "doc.go",
        "docker_container.go",
        "docker_image.go",
        "docker_sandbox.go",
        "docker_service.go",
        "docker_streaming.go",
        "helpers.go",
        "naming.go",
        "security_context.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "convert_test.go",
        "docker_container_test.go",
        "docker_image_test.go",
        "docker_sandbox_test.go",
        "docker_service_test.go",
        "helpers_test.go",
        "naming_test.go",
        "security_context_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
