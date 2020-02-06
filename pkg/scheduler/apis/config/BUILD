load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "legacy_types.go",
        "register.go",
        "types.go",
        "zz_generated.deepcopy.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/scheduler/apis/config",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/component-base/config:go_default_library",
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
        "//pkg/scheduler/apis/config/scheme:all-srcs",
        "//pkg/scheduler/apis/config/testing:all-srcs",
        "//pkg/scheduler/apis/config/v1:all-srcs",
        "//pkg/scheduler/apis/config/v1alpha1:all-srcs",
        "//pkg/scheduler/apis/config/v1alpha2:all-srcs",
        "//pkg/scheduler/apis/config/validation:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["types_test.go"],
    embed = [":go_default_library"],
    deps = ["//vendor/github.com/google/go-cmp/cmp:go_default_library"],
)
