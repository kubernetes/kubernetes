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
        "//staging/src/k8s.io/kubectl/pkg/apply:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/apps:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/cmd:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/describe:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/drain:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/explain:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/generate:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/generated:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/metricsutil:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/polymorphichelpers:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/proxy:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/rawhttp:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/scale:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/scheme:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/validation:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/version:all-srcs",
        "//staging/src/k8s.io/kubectl/test/data:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
