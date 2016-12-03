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
        "config.go",
        "mocksync.go",
        "nopsync.go",
        "sync.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/release_1_5:go_default_library",
        "//pkg/dns/federation:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/client-go/pkg/util/wait",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["config_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor:github.com/stretchr/testify/assert"],
)
