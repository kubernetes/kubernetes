package testing

import (
	"fmt"
)

const FipDNS = `{
        "floating_network_id": "6d67c30a-ddb4-49a1-bec3-a65b286b4170",
        "router_id": null,
        "fixed_ip_address": null,
        "floating_ip_address": "192.0.0.4",
        "tenant_id": "017d8de156df4177889f31a9bd6edc00",
        "status": "DOWN",
        "port_id": null,
        "id": "2f95fd2b-9f6a-4e8e-9e9a-2cbe286cbf9e",
        "router_id": "1117c30a-ddb4-49a1-bec3-a65b286b4170",
        "dns_domain": "local.",
        "dns_name": "test-fip"
    }`

const FipNoDNS = `{
        "floating_network_id": "90f742b1-6d17-487b-ba95-71881dbc0b64",
        "router_id": "0a24cb83-faf5-4d7f-b723-3144ed8a2167",
        "fixed_ip_address": "192.0.0.2",
        "floating_ip_address": "10.0.0.3",
        "tenant_id": "017d8de156df4177889f31a9bd6edc00",
        "status": "DOWN",
        "port_id": "74a342ce-8e07-4e91-880c-9f834b68fa25",
        "id": "ada25a95-f321-4f59-b0e0-f3a970dd3d63",
        "router_id": "2227c30a-ddb4-49a1-bec3-a65b286b4170",
        "dns_domain": "",
        "dns_name": ""
    }`

var ListResponse = fmt.Sprintf(`
{
    "floatingips": [
%s,
%s
    ]
}
`, FipDNS, FipNoDNS)

var ListResponseDNS = fmt.Sprintf(`
{
    "floatingips": [
%s
    ]
}
`, FipDNS)
