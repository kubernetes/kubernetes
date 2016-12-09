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
        "metadata.go",
        "openstack.go",
        "openstack_instances.go",
        "openstack_loadbalancer.go",
        "openstack_routes.go",
        "openstack_volumes.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/resource:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/api/v1/service:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/volume:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack",
        "//vendor:github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes",
        "//vendor:github.com/rackspace/gophercloud/openstack/compute/v2/extensions/volumeattach",
        "//vendor:github.com/rackspace/gophercloud/openstack/compute/v2/flavors",
        "//vendor:github.com/rackspace/gophercloud/openstack/compute/v2/servers",
        "//vendor:github.com/rackspace/gophercloud/openstack/identity/v3/extensions/trust",
        "//vendor:github.com/rackspace/gophercloud/openstack/identity/v3/tokens",
        "//vendor:github.com/rackspace/gophercloud/openstack/networking/v2/extensions",
        "//vendor:github.com/rackspace/gophercloud/openstack/networking/v2/extensions/layer3/floatingips",
        "//vendor:github.com/rackspace/gophercloud/openstack/networking/v2/extensions/layer3/routers",
        "//vendor:github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/members",
        "//vendor:github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/monitors",
        "//vendor:github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/pools",
        "//vendor:github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/vips",
        "//vendor:github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/listeners",
        "//vendor:github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/loadbalancers",
        "//vendor:github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/monitors",
        "//vendor:github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/pools",
        "//vendor:github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/groups",
        "//vendor:github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/rules",
        "//vendor:github.com/rackspace/gophercloud/openstack/networking/v2/ports",
        "//vendor:github.com/rackspace/gophercloud/pagination",
        "//vendor:gopkg.in/gcfg.v1",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "metadata_test.go",
        "openstack_routes_test.go",
        "openstack_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/rand:go_default_library",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack/compute/v2/servers",
    ],
)
