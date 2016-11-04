package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "auth_loaders.go",
        "client_config.go",
        "config.go",
        "doc.go",
        "loader.go",
        "merged_client_builder.go",
        "overrides.go",
        "validation.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/unversioned/auth:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
        "//pkg/client/unversioned/clientcmd/api/latest:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/homedir:go_default_library",
        "//pkg/util/validation:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/howeyc/gopass",
        "//vendor:github.com/imdario/mergo",
        "//vendor:github.com/spf13/pflag",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "client_config_test.go",
        "loader_test.go",
        "merged_client_builder_test.go",
        "validation_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
        "//pkg/client/unversioned/clientcmd/api/latest:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//vendor:github.com/ghodss/yaml",
        "//vendor:github.com/imdario/mergo",
    ],
)
