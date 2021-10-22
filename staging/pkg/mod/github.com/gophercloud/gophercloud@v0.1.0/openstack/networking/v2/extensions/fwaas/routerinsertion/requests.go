package routerinsertion

import (
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/fwaas/firewalls"
)

// CreateOptsExt adds the RouterIDs option to the base CreateOpts.
type CreateOptsExt struct {
	firewalls.CreateOptsBuilder
	RouterIDs []string `json:"router_ids"`
}

// ToFirewallCreateMap adds router_ids to the base firewall creation options.
func (opts CreateOptsExt) ToFirewallCreateMap() (map[string]interface{}, error) {
	base, err := opts.CreateOptsBuilder.ToFirewallCreateMap()
	if err != nil {
		return nil, err
	}

	firewallMap := base["firewall"].(map[string]interface{})
	firewallMap["router_ids"] = opts.RouterIDs

	return base, nil
}

// UpdateOptsExt adds the RouterIDs option to the base UpdateOpts.
type UpdateOptsExt struct {
	firewalls.UpdateOptsBuilder
	RouterIDs []string `json:"router_ids"`
}

// ToFirewallUpdateMap adds router_ids to the base firewall update options.
func (opts UpdateOptsExt) ToFirewallUpdateMap() (map[string]interface{}, error) {
	base, err := opts.UpdateOptsBuilder.ToFirewallUpdateMap()
	if err != nil {
		return nil, err
	}

	firewallMap := base["firewall"].(map[string]interface{})
	firewallMap["router_ids"] = opts.RouterIDs

	return base, nil
}
