load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "build.go",
        "desc.go",
        "desc_init.go",
        "desc_lazy.go",
        "desc_list.go",
        "desc_list_gen.go",
        "placeholder.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/protobuf/internal/filedesc",
    importpath = "google.golang.org/protobuf/internal/filedesc",
    visibility = ["//vendor/google.golang.org/protobuf:__subpackages__"],
    deps = [
        "//vendor/google.golang.org/protobuf/encoding/protowire:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/descfmt:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/descopts:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/encoding/defval:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/errors:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/fieldnum:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/pragma:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/strs:go_default_library",
        "//vendor/google.golang.org/protobuf/proto:go_default_library",
        "//vendor/google.golang.org/protobuf/reflect/protoreflect:go_default_library",
        "//vendor/google.golang.org/protobuf/reflect/protoregistry:go_default_library",
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
