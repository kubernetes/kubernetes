package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["probe_plugins.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/volume:go_default_library",
        "//pkg/volume/aws_ebs:go_default_library",
        "//pkg/volume/azure_dd:go_default_library",
        "//pkg/volume/azure_file:go_default_library",
        "//pkg/volume/cephfs:go_default_library",
        "//pkg/volume/cinder:go_default_library",
        "//pkg/volume/configmap:go_default_library",
        "//pkg/volume/downwardapi:go_default_library",
        "//pkg/volume/empty_dir:go_default_library",
        "//pkg/volume/fc:go_default_library",
        "//pkg/volume/flexvolume:go_default_library",
        "//pkg/volume/flocker:go_default_library",
        "//pkg/volume/gce_pd:go_default_library",
        "//pkg/volume/git_repo:go_default_library",
        "//pkg/volume/glusterfs:go_default_library",
        "//pkg/volume/host_path:go_default_library",
        "//pkg/volume/iscsi:go_default_library",
        "//pkg/volume/nfs:go_default_library",
        "//pkg/volume/photon_pd:go_default_library",
        "//pkg/volume/portworx:go_default_library",
        "//pkg/volume/projected:go_default_library",
        "//pkg/volume/quobyte:go_default_library",
        "//pkg/volume/rbd:go_default_library",
        "//pkg/volume/scaleio:go_default_library",
        "//pkg/volume/secret:go_default_library",
        "//pkg/volume/vsphere_volume:go_default_library",
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
)
