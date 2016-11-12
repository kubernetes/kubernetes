package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_binary(
    name = "hyperkube",
    srcs = [
        "federation-apiserver.go",
        "federation-controller-manager.go",
        "hyperkube.go",
        "kube-apiserver.go",
        "kube-controller-manager.go",
        "kube-proxy.go",
        "kube-scheduler.go",
        "kubectl.go",
        "kubelet.go",
        "main.go",
        "server.go",
    ],
    tags = ["automanaged"],
)
