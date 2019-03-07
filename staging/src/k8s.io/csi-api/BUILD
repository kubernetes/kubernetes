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
        "//staging/src/k8s.io/csi-api/pkg/apis/csi/v1alpha1:all-srcs",
        "//staging/src/k8s.io/csi-api/pkg/client/clientset/versioned:all-srcs",
        "//staging/src/k8s.io/csi-api/pkg/client/informers/externalversions:all-srcs",
        "//staging/src/k8s.io/csi-api/pkg/client/listers/csi/v1alpha1:all-srcs",
        "//staging/src/k8s.io/csi-api/pkg/crd:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
