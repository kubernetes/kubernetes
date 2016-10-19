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
    srcs = ["ssh.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/net:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:golang.org/x/crypto/ssh",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["ssh_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/util/wait:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/crypto/ssh",
    ],
)
