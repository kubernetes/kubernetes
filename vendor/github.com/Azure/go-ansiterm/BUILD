package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

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
        "parser_test_helpers.go",
        "parser_test_utilities.go",
        "states.go",
        "test_event_handler.go",
        "utilities.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor/github.com/Sirupsen/logrus:go_default_library"],
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
)
