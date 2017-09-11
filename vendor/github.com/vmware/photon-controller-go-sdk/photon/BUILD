load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "apitypes.go",
        "auth.go",
        "availabilityzones.go",
        "client.go",
        "deployments.go",
        "disks.go",
        "flavors.go",
        "hosts.go",
        "images.go",
        "info.go",
        "networks.go",
        "projects.go",
        "resourcetickets.go",
        "restclient.go",
        "routers.go",
        "services.go",
        "status.go",
        "subnets.go",
        "tasks.go",
        "tenants.go",
        "util.go",
        "virtualnetworks.go",
        "vms.go",
    ],
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/vmware/photon-controller-go-sdk/photon/lightwave:go_default_library"],
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
        "//vendor/github.com/vmware/photon-controller-go-sdk/photon/lightwave:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
