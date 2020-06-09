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
        "//staging/src/k8s.io/legacy-cloud-providers/aws:all-srcs",
        "//staging/src/k8s.io/legacy-cloud-providers/azure:all-srcs",
        "//staging/src/k8s.io/legacy-cloud-providers/gce:all-srcs",
        "//staging/src/k8s.io/legacy-cloud-providers/openstack:all-srcs",
        "//staging/src/k8s.io/legacy-cloud-providers/vsphere:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
