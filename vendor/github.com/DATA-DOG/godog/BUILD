load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "ast.go",
        "builder.go",
        "builder_go110.go",
        "flags.go",
        "fmt.go",
        "fmt_cucumber.go",
        "fmt_events.go",
        "fmt_junit.go",
        "fmt_pretty.go",
        "fmt_progress.go",
        "gherkin.go",
        "godog.go",
        "options.go",
        "run.go",
        "stacktrace.go",
        "stepdef.go",
        "suite.go",
        "suite_context.go",
        "utils.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/DATA-DOG/godog",
    importpath = "github.com/DATA-DOG/godog",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/DATA-DOG/godog/colors:go_default_library",
        "//vendor/github.com/DATA-DOG/godog/gherkin:go_default_library",
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
        "//vendor/github.com/DATA-DOG/godog/colors:all-srcs",
        "//vendor/github.com/DATA-DOG/godog/gherkin:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
