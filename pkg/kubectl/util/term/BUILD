package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "resize.go",
        "term.go",
        "term_writer.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "resizeevents.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "resizeevents.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "resizeevents.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "resizeevents.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "resizeevents.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "resizeevents.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "resizeevents.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "resizeevents.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "resizeevents.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "resizeevents.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "resizeevents_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/kubectl/util/term",
    deps = [
        "//pkg/util/interrupt:go_default_library",
        "//vendor/github.com/docker/docker/pkg/term:go_default_library",
        "//vendor/github.com/mitchellh/go-wordwrap:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/client-go/tools/remotecommand:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = ["term_writer_test.go"],
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
