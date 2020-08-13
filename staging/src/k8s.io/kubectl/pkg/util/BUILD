load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "apply.go",
        "pod_port.go",
        "service_port.go",
        "umask.go",
        "umask_windows.go",
        "util.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kubectl/pkg/util",
    importpath = "k8s.io/kubectl/pkg/util",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:aix": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
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
        "@io_bazel_rules_go//go/platform:illumos": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:ios": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:js": [
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
    srcs = [
        "pod_port_test.go",
        "service_port_test.go",
        "util_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
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
        "//staging/src/k8s.io/kubectl/pkg/util/certificate:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/deployment:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/event:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/fieldpath:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/hash:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/i18n:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/interrupt:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/logs:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/openapi:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/podutils:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/qos:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/rbac:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/resource:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/slice:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/storage:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/templates:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util/term:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
