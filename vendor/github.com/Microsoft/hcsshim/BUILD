load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "container.go",
        "errors.go",
        "hcsshim.go",
        "hnsendpoint.go",
        "hnsglobals.go",
        "hnsnetwork.go",
        "hnspolicy.go",
        "hnspolicylist.go",
        "hnssupport.go",
        "interface.go",
        "layer.go",
        "process.go",
        "zsyscall_windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/Microsoft/hcsshim",
    importpath = "github.com/Microsoft/hcsshim",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/Microsoft/hcsshim/internal/guid:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/hcs:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/hcserror:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/hns:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/mergemaps:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/schema1:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/wclayer:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/golang.org/x/sys/windows:go_default_library",
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
        "//vendor/github.com/Microsoft/hcsshim/hcn:all-srcs",
        "//vendor/github.com/Microsoft/hcsshim/internal/cni:all-srcs",
        "//vendor/github.com/Microsoft/hcsshim/internal/guid:all-srcs",
        "//vendor/github.com/Microsoft/hcsshim/internal/hcs:all-srcs",
        "//vendor/github.com/Microsoft/hcsshim/internal/hcserror:all-srcs",
        "//vendor/github.com/Microsoft/hcsshim/internal/hns:all-srcs",
        "//vendor/github.com/Microsoft/hcsshim/internal/interop:all-srcs",
        "//vendor/github.com/Microsoft/hcsshim/internal/logfields:all-srcs",
        "//vendor/github.com/Microsoft/hcsshim/internal/longpath:all-srcs",
        "//vendor/github.com/Microsoft/hcsshim/internal/mergemaps:all-srcs",
        "//vendor/github.com/Microsoft/hcsshim/internal/regstate:all-srcs",
        "//vendor/github.com/Microsoft/hcsshim/internal/runhcs:all-srcs",
        "//vendor/github.com/Microsoft/hcsshim/internal/safefile:all-srcs",
        "//vendor/github.com/Microsoft/hcsshim/internal/schema1:all-srcs",
        "//vendor/github.com/Microsoft/hcsshim/internal/schema2:all-srcs",
        "//vendor/github.com/Microsoft/hcsshim/internal/timeout:all-srcs",
        "//vendor/github.com/Microsoft/hcsshim/internal/wclayer:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
