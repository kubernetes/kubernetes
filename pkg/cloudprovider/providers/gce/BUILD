package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
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

go_library(
    name = "gce",
    srcs = [
        "doc.go",
        "gce.go",
        "gce_address_manager.go",
        "gce_addresses.go",
        "gce_addresses_fakes.go",
        "gce_alpha.go",
        "gce_annotations.go",
        "gce_backendservice.go",
        "gce_cert.go",
        "gce_clusterid.go",
        "gce_clusters.go",
        "gce_disks.go",
        "gce_firewall.go",
        "gce_forwardingrule.go",
        "gce_forwardingrule_fakes.go",
        "gce_healthchecks.go",
        "gce_instancegroup.go",
        "gce_instances.go",
        "gce_interfaces.go",
        "gce_loadbalancer.go",
        "gce_loadbalancer_external.go",
        "gce_loadbalancer_internal.go",
        "gce_loadbalancer_naming.go",
        "gce_networkendpointgroup.go",
        "gce_op.go",
        "gce_routes.go",
        "gce_targetpool.go",
        "gce_targetproxy.go",
        "gce_urlmap.go",
        "gce_util.go",
        "gce_zones.go",
        "kms.go",
        "metrics.go",
        "token_source.go",
    ],
)

go_test(
    name = "gce_test",
    size = "small",
    srcs = [
        "gce_address_manager_test.go",
        "gce_annotations_test.go",
        "gce_disks_test.go",
        "gce_healthchecks_test.go",
        "gce_loadbalancer_external_test.go",
        "gce_test.go",
        "metrics_test.go",
    ],
    library = ":gce",
)
