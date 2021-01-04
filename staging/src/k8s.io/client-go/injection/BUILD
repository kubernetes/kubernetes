load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "clients.go",
        "context.go",
        "factories.go",
        "informers.go",
        "interface.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/injection",
    importpath = "k8s.io/client-go/injection",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/client-go/controller:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "clients_test.go",
        "context_test.go",
        "factories_test.go",
        "informers_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/client-go/controller:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
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
