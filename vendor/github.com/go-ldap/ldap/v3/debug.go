package ldap

import (
	ber "github.com/go-asn1-ber/asn1-ber"
)

// debugging type
//     - has a Printf method to write the debug output
type debugging bool

// Enable controls debugging mode.
func (debug *debugging) Enable(b bool) {
	*debug = debugging(b)
}

// Printf writes debug output.
func (debug debugging) Printf(format string, args ...interface{}) {
	if debug {
		logger.Printf(format, args...)
	}
}

// PrintPacket dumps a packet.
func (debug debugging) PrintPacket(packet *ber.Packet) {
	if debug {
		ber.WritePacket(logger.Writer(), packet)
	}
}
