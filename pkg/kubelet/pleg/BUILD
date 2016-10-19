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
        "generic.go",
        "pleg.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/metrics:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/clock:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["generic_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/container/testing:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/clock:go_default_library",
        "//pkg/util/diff:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
    ],
)
