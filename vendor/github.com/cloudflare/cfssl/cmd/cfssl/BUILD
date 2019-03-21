load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["cfssl.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/cloudflare/cfssl/cmd/cfssl",
    importpath = "github.com/cloudflare/cfssl/cmd/cfssl",
    visibility = ["//visibility:private"],
    deps = [
        "//vendor/github.com/cloudflare/cfssl/cli:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/bundle:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/certinfo:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/crl:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/gencert:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/gencrl:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/gencsr:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/genkey:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/info:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/ocspdump:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/ocsprefresh:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/ocspserve:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/ocspsign:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/printdefault:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/revoke:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/scan:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/selfsign:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/serve:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/sign:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/version:go_default_library",
        "//vendor/github.com/go-sql-driver/mysql:go_default_library",
        "//vendor/github.com/lib/pq:go_default_library",
        "//vendor/github.com/mattn/go-sqlite3:go_default_library",
    ],
)

go_binary(
    name = "cfssl",
    embed = [":go_default_library"],
    tags = ["manual"],
    visibility = ["//visibility:public"],
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
