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
        "apiserver.go",
        "common.go",
        "config.go",
        "doc.go",
        "file.go",
        "file_linux.go",
        "http.go",
        "sources.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/validation:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/record:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/events:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/kubelet/util/format:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/config:go_default_library",
        "//pkg/util/hash:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/validation/field:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/util/yaml:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/exp/inotify",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "apiserver_test.go",
        "common_test.go",
        "config_test.go",
        "file_linux_test.go",
        "http_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/api/validation:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/record:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/securitycontext:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/testing:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/watch:go_default_library",
    ],
)
