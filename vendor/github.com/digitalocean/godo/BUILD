package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "account.go",
        "action.go",
        "certificates.go",
        "doc.go",
        "domains.go",
        "droplet_actions.go",
        "droplets.go",
        "errors.go",
        "firewalls.go",
        "floating_ips.go",
        "floating_ips_actions.go",
        "godo.go",
        "image_actions.go",
        "images.go",
        "keys.go",
        "links.go",
        "load_balancers.go",
        "regions.go",
        "sizes.go",
        "snapshots.go",
        "storage.go",
        "storage_actions.go",
        "strings.go",
        "tags.go",
        "timestamp.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/digitalocean/godo/context:go_default_library",
        "//vendor/github.com/google/go-querystring/query:go_default_library",
        "//vendor/github.com/tent/http-link-go:go_default_library",
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
        "//vendor/github.com/digitalocean/godo/context:all-srcs",
        "//vendor/github.com/digitalocean/godo/util:all-srcs",
    ],
    tags = ["automanaged"],
)
