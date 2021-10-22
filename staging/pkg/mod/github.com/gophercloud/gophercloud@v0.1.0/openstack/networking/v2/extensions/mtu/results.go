package mtu

// NetworkMTUExt represents an extended form of a Network with additional MTU field.
type NetworkMTUExt struct {
	// The maximum transmission unit (MTU) value to address fragmentation.
	// Minimum value is 68 for IPv4, and 1280 for IPv6.
	MTU int `json:"mtu"`
}
