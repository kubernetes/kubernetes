load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "activatelayer.go",
        "baselayer.go",
        "callback.go",
        "cgo.go",
        "container.go",
        "createlayer.go",
        "createsandboxlayer.go",
        "deactivatelayer.go",
        "destroylayer.go",
        "errors.go",
        "expandsandboxsize.go",
        "exportlayer.go",
        "getlayermountpath.go",
        "getsharedbaseimages.go",
        "guid.go",
        "hcsshim.go",
        "hnsendpoint.go",
        "hnsfuncs.go",
        "hnsnetwork.go",
        "hnspolicy.go",
        "hnspolicylist.go",
        "importlayer.go",
        "interface.go",
        "layerexists.go",
        "layerutils.go",
        "legacy.go",
        "nametoguid.go",
        "preparelayer.go",
        "process.go",
        "processimage.go",
        "unpreparelayer.go",
        "utils.go",
        "version.go",
        "waithelper.go",
        "zhcsshim.go",
    ],
    cgo = True,
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/Microsoft/go-winio:go_default_library",
        "//vendor/github.com/sirupsen/logrus:go_default_library",
        "//vendor/golang.org/x/sys/windows:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
