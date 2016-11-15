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
        "change.go",
        "changes_create_call.go",
        "changes_service.go",
        "clouddns.go",
        "managed_zone.go",
        "managed_zone_create_call.go",
        "managed_zones_delete_call.go",
        "managed_zones_get_call.go",
        "managed_zones_list_call.go",
        "managed_zones_list_response.go",
        "managed_zones_service.go",
        "rrset.go",
        "rrsets_list_call.go",
        "rrsets_list_response.go",
        "rrsets_service.go",
        "service.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//federation/pkg/dnsprovider/providers/google/clouddns/internal/interfaces:go_default_library",
        "//federation/pkg/dnsprovider/rrstype:go_default_library",
        "//vendor:google.golang.org/api/dns/v1",
        "//vendor:google.golang.org/api/googleapi",
    ],
)
