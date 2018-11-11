load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "buffered.go",
        "doc.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/plugin/pkg/audit/buffered",
    importpath = "k8s.io/apiserver/plugin/pkg/audit/buffered",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/audit:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/audit:go_default_library",
        "//staging/src/k8s.io/client-go/util/flowcontrol:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["buffered_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/audit:go_default_library",
        "//staging/src/k8s.io/apiserver/plugin/pkg/audit/fake:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
