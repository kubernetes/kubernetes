package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["ssh.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:golang.org/x/crypto/ssh",
        "//vendor:k8s.io/client-go/pkg/util/net",
        "//vendor:k8s.io/client-go/pkg/util/runtime",
        "//vendor:k8s.io/client-go/pkg/util/wait",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["ssh_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/crypto/ssh",
        "//vendor:k8s.io/client-go/pkg/util/wait",
    ],
)
