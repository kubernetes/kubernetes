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
        "consul_helper.go",
        "consul_watcher.go",
        "errors.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/storage:go_default_library",
        "//pkg/storage/consul/metrics:go_default_library",
        "//pkg/storage/etcd:go_default_library",
        "//pkg/util:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/hashicorp/consul/api",
        "//vendor:github.com/hashicorp/consul/watch",
        "//vendor:golang.org/x/net/context",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["consul_watcher_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/storage:go_default_library",
        "//pkg/storage/consul/consultest:go_default_library",
        "//pkg/storage/consul/testing:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/hashicorp/consul/api",
        "//vendor:golang.org/x/net/context",
    ],
)
