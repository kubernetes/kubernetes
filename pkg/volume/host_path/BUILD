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
        "host_path.go",
    ],
    deps = [
        "//pkg/volume:go_default_library",
        "//pkg/volume/util/volumehelper:go_default_library",
        "//pkg/volume/validation:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "host_path_test.go",
        ],
        "//conditions:default": [],
    }),
    library = ":go_default_library",
    deps = select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//pkg/util/file:go_default_library",
            "//pkg/volume:go_default_library",
            "//pkg/volume/testing:go_default_library",
            "//vendor/k8s.io/api/core/v1:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
            "//vendor/k8s.io/client-go/kubernetes/fake:go_default_library",
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
