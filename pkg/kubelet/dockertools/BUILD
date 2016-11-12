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
        "container_gc.go",
        "convert.go",
        "docker.go",
        "docker_manager.go",
        "docker_manager_linux.go",
        "exec.go",
        "fake_docker_client.go",
        "fake_manager.go",
        "images.go",
        "instrumented_docker.go",
        "kube_docker_client.go",
        "labels.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "container_gc_test.go",
        "convert_test.go",
        "docker_manager_test.go",
        "docker_test.go",
        "images_test.go",
        "labels_test.go",
    ],
    data = [
        "fixtures/seccomp/sub/subtest",
        "fixtures/seccomp/test",
    ],
    library = "go_default_library",
    tags = [
        "automanaged",
        "skip",
    ],
    deps = [],
)
