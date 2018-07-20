load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

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

go_test(
    name = "go_default_test",
    srcs = [
        "parser_test.go",
        "parser_test_helpers_test.go",
        "parser_test_utilities_test.go",
        "test_event_handler_test.go",
    ],
    embed = [":go_default_library"],
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
