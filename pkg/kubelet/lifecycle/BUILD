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
        "doc.go",
        "fake_handler_runner.go",
        "handlers.go",
        "interfaces.go",
        "predicate.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/kubelet/util/format:go_default_library",
        "//pkg/security/apparmor:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/intstr:go_default_library",
        "//plugin/pkg/scheduler/algorithm/predicates:go_default_library",
        "//plugin/pkg/scheduler/schedulercache:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["handlers_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/util/intstr:go_default_library",
    ],
)
