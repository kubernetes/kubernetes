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
        "//staging/src/k8s.io/code-generator/cmd/client-gen:all-srcs",
        "//staging/src/k8s.io/code-generator/cmd/conversion-gen:all-srcs",
        "//staging/src/k8s.io/code-generator/cmd/deepcopy-gen:all-srcs",
        "//staging/src/k8s.io/code-generator/cmd/defaulter-gen:all-srcs",
        "//staging/src/k8s.io/code-generator/cmd/go-to-protobuf:all-srcs",
        "//staging/src/k8s.io/code-generator/cmd/import-boss:all-srcs",
        "//staging/src/k8s.io/code-generator/cmd/informer-gen:all-srcs",
        "//staging/src/k8s.io/code-generator/cmd/lister-gen:all-srcs",
        "//staging/src/k8s.io/code-generator/cmd/openapi-gen:all-srcs",
        "//staging/src/k8s.io/code-generator/cmd/prerelease-lifecycle-gen:all-srcs",
        "//staging/src/k8s.io/code-generator/cmd/register-gen:all-srcs",
        "//staging/src/k8s.io/code-generator/cmd/set-gen:all-srcs",
        "//staging/src/k8s.io/code-generator/hack:all-srcs",
        "//staging/src/k8s.io/code-generator/pkg/namer:all-srcs",
        "//staging/src/k8s.io/code-generator/pkg/util:all-srcs",
        "//staging/src/k8s.io/code-generator/third_party/forked/golang/reflect:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
