load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "pod_port.go",
        "service_port.go",
        "umask.go",
        "umask_windows.go",
        "util.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubectl/util",
    visibility = ["//build/visible_to:pkg_kubectl_util_CONSUMERS"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
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
        "//pkg/kubectl/util/certificate:all-srcs",
        "//pkg/kubectl/util/deployment:all-srcs",
        "//pkg/kubectl/util/event:all-srcs",
        "//pkg/kubectl/util/fieldpath:all-srcs",
        "//pkg/kubectl/util/hash:all-srcs",
        "//pkg/kubectl/util/i18n:all-srcs",
        "//pkg/kubectl/util/logs:all-srcs",
        "//pkg/kubectl/util/podutils:all-srcs",
        "//pkg/kubectl/util/printers:all-srcs",
        "//pkg/kubectl/util/qos:all-srcs",
        "//pkg/kubectl/util/rbac:all-srcs",
        "//pkg/kubectl/util/resource:all-srcs",
        "//pkg/kubectl/util/slice:all-srcs",
        "//pkg/kubectl/util/storage:all-srcs",
        "//pkg/kubectl/util/templates:all-srcs",
        "//pkg/kubectl/util/term:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//build/visible_to:pkg_kubectl_util_CONSUMERS"],
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
