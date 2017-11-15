load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["discovery.go"],
    importpath = "k8s.io/client-go/discovery/fake",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/version:go_default_library",
        "//rest:go_default_library",
        "//testing:go_default_library",
        "//vendor/github.com/googleapis/gnostic/OpenAPIv2:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/version:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["discovery_test.go"],
    importpath = "k8s.io/client-go/discovery/fake_test",
    deps = [
        ":go_default_library",
        "//kubernetes/fake:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/version:go_default_library",
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
