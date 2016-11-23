package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "rollout.go",
        "rollout_history.go",
        "rollout_pause.go",
        "rollout_resume.go",
        "rollout_status.go",
        "rollout_undo.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/kubectl:go_default_library",
        "//pkg/kubectl/cmd/set:go_default_library",
        "//pkg/kubectl/cmd/templates:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/kubectl/resource:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/interrupt:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/renstrom/dedent",
        "//vendor:github.com/spf13/cobra",
    ],
)
