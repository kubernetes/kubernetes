package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "ciphersuites_flag_test.go",
        "colon_separated_multimap_string_string_test.go",
        "langle_separated_map_string_string_test.go",
        "map_string_bool_test.go",
        "map_string_string_test.go",
        "namedcertkey_flag_test.go",
    ],
    embed = [":go_default_library"],
    deps = ["//vendor/github.com/spf13/pflag:go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "ciphersuites_flag.go",
        "ciphersuites_flag_114.go",
        "colon_separated_multimap_string_string.go",
        "configuration_map.go",
        "flags.go",
        "langle_separated_map_string_string.go",
        "map_string_bool.go",
        "map_string_string.go",
        "namedcertkey_flag.go",
        "noop.go",
        "omitempty.go",
        "sectioned.go",
        "string_flag.go",
        "tristate.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/component-base/cli/flag",
    importpath = "k8s.io/component-base/cli/flag",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
