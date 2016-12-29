package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "mocksync.go",
        "nopsync.go",
        "sync.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/dns/federation:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/client-go/kubernetes",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/pkg/apis/meta/v1",
        "//vendor:k8s.io/client-go/pkg/fields",
        "//vendor:k8s.io/client-go/pkg/runtime",
        "//vendor:k8s.io/client-go/pkg/util/wait",
        "//vendor:k8s.io/client-go/pkg/watch",
        "//vendor:k8s.io/client-go/tools/cache",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["config_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor:github.com/stretchr/testify/assert"],
)
