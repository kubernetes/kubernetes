load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "ast.go",
        "comma_token.go",
        "comment_token.go",
        "doc.go",
        "empty_token.go",
        "expression.go",
        "ini.go",
        "ini_lexer.go",
        "ini_parser.go",
        "literal_tokens.go",
        "newline_token.go",
        "number_helper.go",
        "op_tokens.go",
        "parse_error.go",
        "parse_stack.go",
        "sep_tokens.go",
        "skipper.go",
        "statement.go",
        "value_util.go",
        "visitor.go",
        "walker.go",
        "ws_token.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/aws/aws-sdk-go/internal/ini",
    importpath = "github.com/aws/aws-sdk-go/internal/ini",
    visibility = ["//vendor/github.com/aws/aws-sdk-go:__subpackages__"],
    deps = ["//vendor/github.com/aws/aws-sdk-go/aws/awserr:go_default_library"],
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
