load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "apiserver.go",
        "common.go",
        "config.go",
        "defaults.go",
        "doc.go",
        "file.go",
        "flags.go",
        "http.go",
        "sources.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "file_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "file_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "file_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "file_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "file_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "file_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "file_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "file_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "file_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "file_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "file_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/kubelet/config",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//pkg/apis/core/helper:go_default_library",
        "//pkg/apis/core/install:go_default_library",
        "//pkg/apis/core/v1:go_default_library",
        "//pkg/apis/core/validation:go_default_library",
        "//pkg/kubelet/checkpoint:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/events:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/kubelet/util/format:go_default_library",
        "//pkg/util/config:go_default_library",
        "//pkg/util/hash:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/yaml:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/k8s.io/client-go/tools/record:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/exp/inotify:go_default_library",
            "//vendor/k8s.io/client-go/util/flowcontrol:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = [
        "apiserver_test.go",
        "common_test.go",
        "config_test.go",
        "file_test.go",
        "http_test.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "file_linux_test.go",
        ],
        "//conditions:default": [],
    }),
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/pkg/kubelet/config",
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//pkg/apis/core/v1:go_default_library",
        "//pkg/apis/core/validation:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/securitycontext:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/scheme:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/k8s.io/client-go/tools/record:go_default_library",
        "//vendor/k8s.io/client-go/util/testing:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
