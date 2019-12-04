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
        "empty_dir.go",
        "empty_dir_linux.go",
        "empty_dir_unsupported.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/volume/emptydir",
    deps = [
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//pkg/volume/util/fsquota:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/k8s.io/utils/mount:go_default_library",
        "//vendor/k8s.io/utils/strings:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = ["empty_dir_test.go"],
    embed = [":go_default_library"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//pkg/volume:go_default_library",
            "//pkg/volume/testing:go_default_library",
            "//pkg/volume/util:go_default_library",
            "//staging/src/k8s.io/api/core/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
            "//staging/src/k8s.io/client-go/util/testing:go_default_library",
            "//vendor/k8s.io/utils/mount:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/volume:go_default_library",
            "//pkg/volume/testing:go_default_library",
            "//pkg/volume/util:go_default_library",
            "//staging/src/k8s.io/api/core/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
            "//staging/src/k8s.io/client-go/util/testing:go_default_library",
            "//vendor/k8s.io/utils/mount:go_default_library",
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
