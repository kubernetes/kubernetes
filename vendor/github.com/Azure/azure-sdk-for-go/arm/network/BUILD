load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "applicationgateways.go",
        "bgpservicecommunities.go",
        "client.go",
        "expressroutecircuitauthorizations.go",
        "expressroutecircuitpeerings.go",
        "expressroutecircuits.go",
        "expressrouteserviceproviders.go",
        "interfaces.go",
        "loadbalancers.go",
        "localnetworkgateways.go",
        "models.go",
        "packetcaptures.go",
        "publicipaddresses.go",
        "routefilterrules.go",
        "routefilters.go",
        "routes.go",
        "routetables.go",
        "securitygroups.go",
        "securityrules.go",
        "subnets.go",
        "usages.go",
        "version.go",
        "virtualnetworkgatewayconnections.go",
        "virtualnetworkgateways.go",
        "virtualnetworkpeerings.go",
        "virtualnetworks.go",
        "watchers.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/Azure/go-autorest/autorest:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest/azure:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest/date:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest/to:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest/validation:go_default_library",
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
