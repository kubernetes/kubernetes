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
        "//staging/src/k8s.io/cluster-bootstrap/token/api:all-srcs",
        "//staging/src/k8s.io/cluster-bootstrap/token/jws:all-srcs",
        "//staging/src/k8s.io/cluster-bootstrap/token/util:all-srcs",
        "//staging/src/k8s.io/cluster-bootstrap/util/secrets:all-srcs",
        "//staging/src/k8s.io/cluster-bootstrap/util/tokens:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
