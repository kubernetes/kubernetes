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
        "api_object_versioner.go",
        "doc.go",
        "etcd_helper.go",
        "etcd_watcher.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/storage:go_default_library",
        "//pkg/storage/etcd/metrics:go_default_library",
        "//pkg/storage/etcd/util:go_default_library",
        "//pkg/util:go_default_library",
        "//pkg/util/cache:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/coreos/etcd/client",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/context",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "api_object_versioner_test.go",
        "etcd_helper_test.go",
        "etcd_watcher_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/testing:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/serializer:go_default_library",
        "//pkg/storage:go_default_library",
        "//pkg/storage/etcd/etcdtest:go_default_library",
        "//pkg/storage/etcd/testing:go_default_library",
        "//pkg/storage/testing:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/coreos/etcd/client",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:golang.org/x/net/context",
    ],
)
