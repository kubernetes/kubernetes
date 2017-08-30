load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth.go",
        "auth_token.go",
        "client.go",
        "help.go",
        "logical.go",
        "renewer.go",
        "request.go",
        "response.go",
        "secret.go",
        "ssh.go",
        "ssh_agent.go",
        "sys.go",
        "sys_audit.go",
        "sys_auth.go",
        "sys_capabilities.go",
        "sys_config_cors.go",
        "sys_generate_root.go",
        "sys_health.go",
        "sys_init.go",
        "sys_leader.go",
        "sys_leases.go",
        "sys_mounts.go",
        "sys_policy.go",
        "sys_rekey.go",
        "sys_rotate.go",
        "sys_seal.go",
        "sys_stepdown.go",
    ],
    importpath = "github.com/hashicorp/vault/api",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/fatih/structs:go_default_library",
        "//vendor/github.com/hashicorp/go-cleanhttp:go_default_library",
        "//vendor/github.com/hashicorp/go-multierror:go_default_library",
        "//vendor/github.com/hashicorp/go-rootcerts:go_default_library",
        "//vendor/github.com/hashicorp/hcl:go_default_library",
        "//vendor/github.com/hashicorp/hcl/hcl/ast:go_default_library",
        "//vendor/github.com/hashicorp/vault/helper/jsonutil:go_default_library",
        "//vendor/github.com/hashicorp/vault/helper/parseutil:go_default_library",
        "//vendor/github.com/mitchellh/mapstructure:go_default_library",
        "//vendor/github.com/sethgrid/pester:go_default_library",
        "//vendor/golang.org/x/net/http2:go_default_library",
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
