load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "local.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/volume/local",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/events:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//pkg/volume/util/hostutil:go_default_library",
        "//pkg/volume/validation:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
        "//vendor/k8s.io/utils/keymutex:go_default_library",
        "//vendor/k8s.io/utils/mount:go_default_library",
        "//vendor/k8s.io/utils/strings:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "local_linux_test.go",
        "local_test.go",
    ],
    embed = [":go_default_library"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//pkg/volume:go_default_library",
            "//pkg/volume/testing:go_default_library",
            "//pkg/volume/util/hostutil:go_default_library",
            "//staging/src/k8s.io/api/core/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
            "//staging/src/k8s.io/client-go/util/testing:go_default_library",
            "//vendor/k8s.io/utils/mount:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//pkg/volume:go_default_library",
            "//pkg/volume/testing:go_default_library",
            "//pkg/volume/util/hostutil:go_default_library",
            "//staging/src/k8s.io/api/core/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
            "//staging/src/k8s.io/client-go/util/testing:go_default_library",
            "//vendor/k8s.io/utils/mount:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:ios": [
            "//pkg/volume:go_default_library",
            "//pkg/volume/testing:go_default_library",
            "//pkg/volume/util/hostutil:go_default_library",
            "//staging/src/k8s.io/api/core/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
            "//staging/src/k8s.io/client-go/util/testing:go_default_library",
            "//vendor/k8s.io/utils/mount:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/volume:go_default_library",
            "//pkg/volume/testing:go_default_library",
            "//pkg/volume/util/hostutil:go_default_library",
            "//staging/src/k8s.io/api/core/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
            "//staging/src/k8s.io/client-go/util/testing:go_default_library",
            "//vendor/k8s.io/utils/mount:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//pkg/volume:go_default_library",
            "//pkg/volume/testing:go_default_library",
            "//pkg/volume/util/hostutil:go_default_library",
            "//staging/src/k8s.io/api/core/v1:go_default_library",
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
    visibility = ["//visibility:public"],
)
