load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "appendblob.go",
        "authorization.go",
        "blob.go",
        "blobsasuri.go",
        "blobserviceclient.go",
        "blockblob.go",
        "client.go",
        "container.go",
        "copyblob.go",
        "directory.go",
        "entity.go",
        "file.go",
        "fileserviceclient.go",
        "leaseblob.go",
        "message.go",
        "odata.go",
        "pageblob.go",
        "queue.go",
        "queueserviceclient.go",
        "share.go",
        "storagepolicy.go",
        "storageservice.go",
        "table.go",
        "table_batch.go",
        "tableserviceclient.go",
        "util.go",
        "version.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/Azure/go-autorest/autorest:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest/azure:go_default_library",
        "//vendor/github.com/satori/uuid:go_default_library",
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
    visibility = ["//visibility:public"],
)
