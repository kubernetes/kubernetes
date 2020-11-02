load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bootstrap_command.go",
        "build_command.go",
        "convert_command.go",
        "generate_command.go",
        "help_command.go",
        "main.go",
        "nodot_command.go",
        "notifications.go",
        "run_command.go",
        "run_watch_and_build_command_flags.go",
        "suite_runner.go",
        "unfocus_command.go",
        "version_command.go",
        "watch_command.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/onsi/ginkgo/ginkgo",
    importpath = "github.com/onsi/ginkgo/ginkgo",
    visibility = ["//visibility:private"],
    deps = [
        "//vendor/github.com/onsi/ginkgo/config:go_default_library",
        "//vendor/github.com/onsi/ginkgo/ginkgo/convert:go_default_library",
        "//vendor/github.com/onsi/ginkgo/ginkgo/interrupthandler:go_default_library",
        "//vendor/github.com/onsi/ginkgo/ginkgo/nodot:go_default_library",
        "//vendor/github.com/onsi/ginkgo/ginkgo/testrunner:go_default_library",
        "//vendor/github.com/onsi/ginkgo/ginkgo/testsuite:go_default_library",
        "//vendor/github.com/onsi/ginkgo/ginkgo/watch:go_default_library",
        "//vendor/github.com/onsi/ginkgo/reporters/stenographer/support/go-colorable:go_default_library",
        "//vendor/github.com/onsi/ginkgo/types:go_default_library",
    ],
)

go_binary(
    name = "ginkgo",
    embed = [":go_default_library"],
    tags = ["manual"],
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
        "//vendor/github.com/onsi/ginkgo/ginkgo/convert:all-srcs",
        "//vendor/github.com/onsi/ginkgo/ginkgo/interrupthandler:all-srcs",
        "//vendor/github.com/onsi/ginkgo/ginkgo/nodot:all-srcs",
        "//vendor/github.com/onsi/ginkgo/ginkgo/testrunner:all-srcs",
        "//vendor/github.com/onsi/ginkgo/ginkgo/testsuite:all-srcs",
        "//vendor/github.com/onsi/ginkgo/ginkgo/watch:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
