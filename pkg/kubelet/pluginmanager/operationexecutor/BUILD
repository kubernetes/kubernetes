load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "operation_executor.go",
        "operation_generator.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/pluginmanager/operationexecutor",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/apis/pluginregistration/v1:go_default_library",
        "//pkg/kubelet/pluginmanager/cache:go_default_library",
        "//pkg/util/goroutinemap:go_default_library",
        "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["operation_executor_test.go"],
    embed = [":go_default_library"],
    deps = ["//pkg/kubelet/pluginmanager/cache:go_default_library"],
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
