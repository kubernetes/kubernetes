load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "flags.go",
        "util.go",
        "viper.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/spf13/viper",
    importpath = "github.com/spf13/viper",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/fsnotify/fsnotify:go_default_library",
        "//vendor/github.com/hashicorp/hcl:go_default_library",
        "//vendor/github.com/hashicorp/hcl/hcl/printer:go_default_library",
        "//vendor/github.com/magiconair/properties:go_default_library",
        "//vendor/github.com/mitchellh/mapstructure:go_default_library",
        "//vendor/github.com/pelletier/go-toml:go_default_library",
        "//vendor/github.com/spf13/afero:go_default_library",
        "//vendor/github.com/spf13/cast:go_default_library",
        "//vendor/github.com/spf13/jwalterweatherman:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/github.com/subosito/gotenv:go_default_library",
        "//vendor/gopkg.in/ini.v1:go_default_library",
        "//vendor/gopkg.in/yaml.v2:go_default_library",
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
    visibility = ["//visibility:public"],
)
