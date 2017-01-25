package subnets

import "fmt"

func err(str string) error {
	return fmt.Errorf("%s", str)
}

var (
	errNetworkIDRequired    = err("A network ID is required")
	errCIDRRequired         = err("A valid CIDR is required")
	errInvalidIPType        = err("An IP type must either be 4 or 6")
	errInvalidGatewayConfig = err("Both disabling the gateway and specifying a gateway is not allowed")
)
