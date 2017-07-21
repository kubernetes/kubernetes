package water

import (
	"errors"
	"io"
)

// Interface is a TUN/TAP interface.
type Interface struct {
	isTAP bool
	io.ReadWriteCloser
	name string
}

// DeviceType is the type for specifying device types.
type DeviceType int

// TUN and TAP device types.
const (
	_ = iota
	TUN
	TAP
)

// Config defines parameters required to create a TUN/TAP interface. It's only
// used when the device is initialized. A zero-value Config is a valid
// configuration.
type Config struct {
	// DeviceType specifies whether the device is a TUN or TAP interface. A
	// zero-value is treated as TUN.
	DeviceType DeviceType

	// PlatformSpecificParams defines parameters that differ on different
	// platforms. See comments for the type for more details.
	PlatformSpecificParams
}

func defaultConfig() Config {
	return Config{
		DeviceType:             TUN,
		PlatformSpecificParams: defaultPlatformSpecificParams(),
	}
}

var zeroConfig Config

// New creates a new TUN/TAP interface using config.
func New(config Config) (ifce *Interface, err error) {
	if zeroConfig == config {
		config = defaultConfig()
	}
	switch config.DeviceType {
	case TUN:
		return newTUN(config)
	case TAP:
		return newTAP(config)
	default:
		return nil, errors.New("unknown device type")
	}
}

// IsTUN returns true if ifce is a TUN interface.
func (ifce *Interface) IsTUN() bool {
	return !ifce.isTAP
}

// IsTAP returns true if ifce is a TAP interface.
func (ifce *Interface) IsTAP() bool {
	return ifce.isTAP
}

// Name returns the interface name of ifce, e.g. tun0, tap1, tun0, etc..
func (ifce *Interface) Name() string {
	return ifce.name
}
