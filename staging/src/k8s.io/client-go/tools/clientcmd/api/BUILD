package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "helpers_test.go",
        "types_test.go",
    ],
    embed = [":go_default_library"],
    deps = ["//vendor/sigs.k8s.io/yaml:go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "helpers.go",
        "register.go",
        "types.go",
        "zz_generated.deepcopy.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/tools/clientcmd/api",
    importpath = "k8s.io/client-go/tools/clientcmd/api",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/client-go/tools/clientcmd/api/latest:all-srcs",
        "//staging/src/k8s.io/client-go/tools/clientcmd/api/v1:all-srcs",
    ],
    tags = ["automanaged"],
)
