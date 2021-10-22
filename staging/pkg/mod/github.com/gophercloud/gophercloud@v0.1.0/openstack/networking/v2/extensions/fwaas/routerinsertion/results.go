package routerinsertion

// FirewallExt is an extension to the base Firewall object
type FirewallExt struct {
	// RouterIDs are the routers that the firewall is attached to.
	RouterIDs []string `json:"router_ids"`
}
