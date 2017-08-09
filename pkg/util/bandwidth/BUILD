package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fake_shaper.go",
        "interfaces.go",
        "unsupported.go",
        "utils.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "linux.go",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = [
        "utils_test.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "linux_test.go",
        ],
        "//conditions:default": [],
    }),
    library = ":go_default_library",
    deps = [
        "//pkg/api:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//vendor/k8s.io/utils/exec:go_default_library",
            "//vendor/k8s.io/utils/exec/testing:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
)
