load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "array.go",
        "config.go",
        "doc.go",
        "encoder.go",
        "error.go",
        "field.go",
        "flag.go",
        "global.go",
        "http_handler.go",
        "level.go",
        "logger.go",
        "options.go",
        "sink.go",
        "stacktrace.go",
        "sugar.go",
        "time.go",
        "writer.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.uber.org/zap",
    importpath = "go.uber.org/zap",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.uber.org/atomic:go_default_library",
        "//vendor/go.uber.org/multierr:go_default_library",
        "//vendor/go.uber.org/zap/internal/bufferpool:go_default_library",
        "//vendor/go.uber.org/zap/zapcore:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/go.uber.org/zap/buffer:all-srcs",
        "//vendor/go.uber.org/zap/internal/bufferpool:all-srcs",
        "//vendor/go.uber.org/zap/internal/color:all-srcs",
        "//vendor/go.uber.org/zap/internal/exit:all-srcs",
        "//vendor/go.uber.org/zap/zapcore:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
