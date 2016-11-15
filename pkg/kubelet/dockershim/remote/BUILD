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
        "docker_server.go",
        "docker_service.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/kubelet/api:go_default_library",
        "//pkg/kubelet/api/v1alpha1/runtime:go_default_library",
        "//pkg/kubelet/dockershim:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/interrupt:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/context",
        "//vendor:google.golang.org/grpc",
    ],
)
