package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "ipvs_linux_test.go",
        "ipvs_test.go",
        "kernelcheck_linux_test.go",
    ],
    embed = [":go_default_library"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/docker/libnetwork/ipvs:go_default_library",
            "//vendor/k8s.io/utils/exec:go_default_library",
            "//vendor/k8s.io/utils/exec/testing:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_library(
    name = "go_default_library",
    srcs = [
        "ipvs.go",
        "ipvs_linux.go",
        "ipvs_unsupported.go",
        "kernelcheck_linux.go",
        "kernelcheck_unsupported.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/util/ipvs",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/version:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
            "//vendor/github.com/docker/libnetwork/ipvs:go_default_library",
            "//vendor/github.com/lithammer/dedent:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//pkg/util/ipvs/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
