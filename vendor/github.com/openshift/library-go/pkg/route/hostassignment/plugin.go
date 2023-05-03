package hostassignment

import (
	"fmt"
	"strings"

	kvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/klog/v2"

	routev1 "github.com/openshift/api/route/v1"
)

// Default DNS suffix to use if no configuration is passed to this plugin.
const defaultDNSSuffix = "router.default.svc.cluster.local"

// SimpleAllocationPlugin implements the route.AllocationPlugin interface
// to provide a simple unsharded (or single sharded) allocation plugin.
type SimpleAllocationPlugin struct {
	DNSSuffix string
}

// NewSimpleAllocationPlugin creates a new SimpleAllocationPlugin.
func NewSimpleAllocationPlugin(suffix string) (*SimpleAllocationPlugin, error) {
	if len(suffix) == 0 {
		suffix = defaultDNSSuffix
	}

	klog.V(4).Infof("Route plugin initialized with suffix=%s", suffix)

	// Check that the DNS suffix is valid.
	if len(kvalidation.IsDNS1123Subdomain(suffix)) != 0 {
		return nil, fmt.Errorf("invalid DNS suffix: %s", suffix)
	}

	return &SimpleAllocationPlugin{DNSSuffix: suffix}, nil
}

// GenerateHostname generates a host name for a route - using the service name,
// namespace (if provided) and the router shard dns suffix.
// TODO: move to router code, and have the routers set this back on the route status.
func (p *SimpleAllocationPlugin) GenerateHostname(route *routev1.Route) (string, error) {
	if len(route.Name) == 0 || len(route.Namespace) == 0 {
		return "", nil
	}
	return fmt.Sprintf("%s-%s.%s", strings.Replace(route.Name, ".", "-", -1), route.Namespace, p.DNSSuffix), nil
}
