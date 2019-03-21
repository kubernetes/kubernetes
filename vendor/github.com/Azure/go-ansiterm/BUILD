load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "constants.go",
        "context.go",
        "csi_entry_state.go",
        "csi_param_state.go",
        "escape_intermediate_state.go",
        "escape_state.go",
        "event_handler.go",
        "ground_state.go",
        "osc_string_state.go",
        "parser.go",
        "parser_action_helpers.go",
        "parser_actions.go",
        "states.go",
        "utilities.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/Azure/go-ansiterm",
    importpath = "github.com/Azure/go-ansiterm",
    visibility = ["//visibility:public"],
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
        "//vendor/github.com/Azure/go-ansiterm/winterm:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
