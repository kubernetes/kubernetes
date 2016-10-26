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
    srcs = ["pod_helper.go"],
    tags = ["automanaged"],
    deps = [
        "//federation/pkg/federation-controller/util:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/labels:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["pod_helper_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//federation/pkg/federation-controller/util:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
    ],
)
