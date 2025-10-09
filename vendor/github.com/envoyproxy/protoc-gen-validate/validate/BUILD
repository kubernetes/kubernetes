load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@io_bazel_rules_go//go:def.bzl", "go_library")
load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_java//java:defs.bzl", "java_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_python//python:proto.bzl", "py_proto_library")

package(
    default_visibility =
        ["//visibility:public"],
)

proto_library(
    name = "validate_proto",
    srcs = ["validate.proto"],
    deps = [
        "@com_google_protobuf//:descriptor_proto",
        "@com_google_protobuf//:duration_proto",
        "@com_google_protobuf//:timestamp_proto",
    ],
)

cc_proto_library(
    name = "validate_cc",
    deps = [":validate_proto"],
)

py_proto_library(
    name = "validate_py",
    deps = [":validate_proto"],
)

go_proto_library(
    name = "validate_go_proto",
    importpath = "github.com/envoyproxy/protoc-gen-validate/validate",
    proto = ":validate_proto",
)

cc_library(
    name = "cc_validate",
    hdrs = ["validate.h"],
)

go_library(
    name = "validate_go",
    embed = [":validate_go_proto"],
    importpath = "github.com/envoyproxy/protoc-gen-validate/validate",
)

java_proto_library(
    name = "validate_java",
    deps = [":validate_proto"],
)

filegroup(
    name = "validate_src",
    srcs = ["validate.proto"],
)

alias(
    name = "go_default_library",
    actual = ":validate",
    deprecation = "Use :validate instead of :go_default_library.  Details about the new naming convention: https://github.com/bazelbuild/bazel-gazelle/pull/863",
    visibility = ["//visibility:public"],
)

# this alias allows build files generated with Gazelle in other repositories
# to find validate as an external dependency
alias(
    name = "validate",
    actual = ":validate_go",
    visibility = ["//visibility:public"],
)
