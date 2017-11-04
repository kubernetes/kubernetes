package(default_visibility = ["//visibility:public"])

licenses(["notice"])

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
        "//third_party/forked/etcd221/pkg/fileutil:all-srcs",
        "//third_party/forked/etcd221/wal:all-srcs",
        "//third_party/forked/etcd237/pkg/fileutil:all-srcs",
        "//third_party/forked/etcd237/wal:all-srcs",
        "//third_party/forked/golang/expansion:all-srcs",
        "//third_party/forked/golang/reflect:all-srcs",
        "//third_party/forked/golang/template:all-srcs",
        "//third_party/forked/gonum/graph:all-srcs",
        "//third_party/htpasswd:all-srcs",
    ],
    tags = ["automanaged"],
)
