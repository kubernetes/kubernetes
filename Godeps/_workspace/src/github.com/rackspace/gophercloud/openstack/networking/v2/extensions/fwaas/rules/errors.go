package rules

import "fmt"

func err(str string) error {
	return fmt.Errorf("%s", str)
}

var (
	errProtocolRequired = err("A protocol is required (tcp, udp, icmp or any)")
	errActionRequired   = err("An action is required (allow or deny)")
)
