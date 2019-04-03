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
        "//staging/src/k8s.io/apiextensions-apiserver/examples/client-go/hack:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/examples/client-go/pkg/apis/cr:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/examples/client-go/pkg/client/clientset/versioned:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/examples/client-go/pkg/client/informers/externalversions:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/examples/client-go/pkg/client/listers/cr/v1:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
