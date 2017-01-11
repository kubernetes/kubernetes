package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "auth_loaders.go",
        "client_config.go",
        "config.go",
        "doc.go",
        "helpers.go",
        "loader.go",
        "merged_client_builder.go",
        "overrides.go",
        "validation.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/unversioned/auth:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
        "//pkg/client/unversioned/clientcmd/api/latest:go_default_library",
        "//pkg/util/homedir:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/howeyc/gopass",
        "//vendor:github.com/imdario/mergo",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/validation",
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
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
        "//pkg/client/unversioned/clientcmd/api/latest:go_default_library",
        "//vendor:github.com/ghodss/yaml",
        "//vendor:github.com/imdario/mergo",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
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
        "//pkg/client/unversioned/clientcmd/api:all-srcs",
    ],
    tags = ["automanaged"],
)
