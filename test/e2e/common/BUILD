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
        "configmap.go",
        "container_probe.go",
        "docker_containers.go",
        "downward_api.go",
        "downwardapi_volume.go",
        "empty_dir.go",
        "expansion.go",
        "host_path.go",
        "init_container.go",
        "kubelet_etc_hosts.go",
        "networking.go",
        "pods.go",
        "privileged.go",
        "secrets.go",
        "sysctl.go",
        "util.go",
        "volumes.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/ginkgo",
        "//vendor:github.com/onsi/gomega",
    ],
)
