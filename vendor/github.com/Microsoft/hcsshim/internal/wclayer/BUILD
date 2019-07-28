load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "activatelayer.go",
        "baselayer.go",
        "createlayer.go",
        "createscratchlayer.go",
        "deactivatelayer.go",
        "destroylayer.go",
        "expandscratchsize.go",
        "exportlayer.go",
        "getlayermountpath.go",
        "getsharedbaseimages.go",
        "grantvmaccess.go",
        "importlayer.go",
        "layerexists.go",
        "layerid.go",
        "layerutils.go",
        "legacy.go",
        "nametoguid.go",
        "preparelayer.go",
        "processimage.go",
        "unpreparelayer.go",
        "wclayer.go",
        "zsyscall_windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/Microsoft/hcsshim/internal/wclayer",
    importpath = "github.com/Microsoft/hcsshim/internal/wclayer",
    visibility = ["//vendor/github.com/Microsoft/hcsshim:__subpackages__"],
    deps = [
        "//vendor/github.com/Microsoft/go-winio:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/guid:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/hcserror:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/interop:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/longpath:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/safefile:go_default_library",
        "//vendor/github.com/sirupsen/logrus:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
