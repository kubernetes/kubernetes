package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "client_config_test.go",
        "loader_test.go",
        "merged_client_builder_test.go",
        "overrides_test.go",
        "validation_test.go",
    ],
    library = ":go_default_library",
    deps = [
        "//vendor/github.com/ghodss/yaml:go_default_library",
        "//vendor/github.com/imdario/mergo:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/tools/clientcmd/api:go_default_library",
        "//vendor/k8s.io/client-go/tools/clientcmd/api/latest:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "auth_loaders.go",
        "client_config.go",
        "config.go",
        "doc.go",
        "flag.go",
        "helpers.go",
        "loader.go",
        "merged_client_builder.go",
        "overrides.go",
        "validation.go",
    ],
    deps = [
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/howeyc/gopass:go_default_library",
        "//vendor/github.com/imdario/mergo:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/validation:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/tools/auth:go_default_library",
        "//vendor/k8s.io/client-go/tools/clientcmd/api:go_default_library",
        "//vendor/k8s.io/client-go/tools/clientcmd/api/latest:go_default_library",
        "//vendor/k8s.io/client-go/util/homedir:go_default_library",
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
        "//staging/src/k8s.io/client-go/tools/clientcmd/api:all-srcs",
    ],
    tags = ["automanaged"],
)
