# doc.go is managed by kazel, so gazelle should ignore it.
# gazelle:exclude doc.go

package(default_visibility = ["//visibility:public"])

load("//pkg/generated/openapi:def.bzl", "openapi_library")
load("//build:openapi.bzl", "openapi_go_prefix", "openapi_vendor_prefix")

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
