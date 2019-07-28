load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "console_encoder.go",
        "core.go",
        "doc.go",
        "encoder.go",
        "entry.go",
        "error.go",
        "field.go",
        "hook.go",
        "json_encoder.go",
        "level.go",
        "level_strings.go",
        "marshaler.go",
        "memory_encoder.go",
        "sampler.go",
        "tee.go",
        "write_syncer.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.uber.org/zap/zapcore",
    importpath = "go.uber.org/zap/zapcore",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.uber.org/atomic:go_default_library",
        "//vendor/go.uber.org/multierr:go_default_library",
        "//vendor/go.uber.org/zap/buffer:go_default_library",
        "//vendor/go.uber.org/zap/internal/bufferpool:go_default_library",
        "//vendor/go.uber.org/zap/internal/color:go_default_library",
        "//vendor/go.uber.org/zap/internal/exit:go_default_library",
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
