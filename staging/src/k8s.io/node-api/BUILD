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
        "//staging/src/k8s.io/node-api/hack:all-srcs",
        "//staging/src/k8s.io/node-api/pkg/apis/node/v1alpha1:all-srcs",
        "//staging/src/k8s.io/node-api/pkg/client/clientset/versioned:all-srcs",
        "//staging/src/k8s.io/node-api/pkg/client/informers/externalversions:all-srcs",
        "//staging/src/k8s.io/node-api/pkg/client/listers/node/v1alpha1:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
