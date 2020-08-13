package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "register.go",
        "types.go",
        "zz_generated.deepcopy.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/apis/extensions",
    deps = [
        "//pkg/apis/apps:go_default_library",
        "//pkg/apis/autoscaling:go_default_library",
        "//pkg/apis/networking:go_default_library",
        "//pkg/apis/policy:go_default_library",
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
        "//pkg/apis/extensions/fuzzer:all-srcs",
        "//pkg/apis/extensions/install:all-srcs",
        "//pkg/apis/extensions/v1beta1:all-srcs",
        "//pkg/apis/extensions/validation:all-srcs",
    ],
    tags = ["automanaged"],
)
