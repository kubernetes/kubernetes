package dhcp4

import (
	"bytes"
	"net"
	"testing"
	"time"
)

func TestNewPacket(t *testing.T) {
	var tests = []struct {
		description string
		opCode      OpCode
	}{
		{
			description: "boot request",
			opCode:      BootRequest,
		},
		{
			description: "boot reply",
			opCode:      BootReply,
		},
		{
			description: "unknown opcode",
			opCode:      3,
		},
	}

	for i, tt := range tests {
		if want, got := newPacket(tt.opCode), NewPacket(tt.opCode); !bytes.Equal(want, got) {
			t.Fatalf("%02d: NewPacket(%d), test %q, unexpected result: %v != %v",
				i, tt.opCode, tt.description, want, got)
		}
	}
}

func TestPacketAddOption(t *testing.T) {
	for i, tt := range optionsTests {
		// Set up new packet, apply options from slice
		p := NewPacket(BootRequest)
		for _, o := range tt.options {
			p.AddOption(o.Code, o.Value)
		}

		// Empty options should result in no changes
		if tt.options == nil || len(tt.options) == 0 {
			if !bytes.Equal(p, NewPacket(BootRequest)) {
				t.Fatalf("%02d: test %q, no options applied, but packet contained extra data",
					i, tt.description)
			}
		}

		// Check that each option was properly applied, in order

		// Track length of previous option bytes
		var offset int
		for ii, o := range tt.options {
			// Options start at byte 240 on an empty packet, adding
			// offset as loops continue
			start := offset + 240
			end := start + 2 + len(o.Value)

			// Options bytes: [option] [length] [value...]
			check := append([]byte{byte(o.Code)}, byte(len(o.Value)))
			check = append(check, o.Value...)

			// Verify option correctly applied
			if want, got := p[start:end], check; !bytes.Equal(want, got) {
				t.Fatalf("%02d: test %q, unexpected option bytes: %v != %v",
					ii, tt.description, want, got)
			}

			// Track offset for next loop
			offset = offset + len(check)
		}

		// Ensure last byte is always End
		if p[len(p)-1] != byte(End) {
			t.Fatalf("%02d: test %q, missing End byte", i, tt.description)
		}
	}
}

func TestPacketParseOptions(t *testing.T) {
	for i, tt := range optionsTests {
		// Set up new packet, apply options from slice
		p := NewPacket(BootRequest)
		for _, o := range tt.options {
			p.AddOption(o.Code, o.Value)
		}

		// Parse options, verify all options are present
		options := p.ParseOptions()
		for _, o := range tt.options {
			var found bool

			// Search for expected option in result map
			for k, v := range options {
				if o.Code == k && bytes.Equal(o.Value, v) {
					found = true
					break
				}
			}

			// Pad option is not parsed, but check all others
			if !found && o.Code != Pad {
				t.Fatalf("%02d: test %q, did not find option: %v",
					i, tt.description, o)
			}
		}
	}
}

func TestPacketStripOptions(t *testing.T) {
	for i, tt := range optionsTests {
		// Set up new packet, apply options from slice
		p := NewPacket(BootRequest)
		for _, o := range tt.options {
			p.AddOption(o.Code, o.Value)
		}

		// Strip all options, verify options are gone
		p.StripOptions()
		if !bytes.Equal(p, NewPacket(BootRequest)) {
			t.Fatalf("%02d: test %q, options stripped, but packet contained extra data",
				i, tt.description)
		}
	}
}

func TestPacketPadToMinSize(t *testing.T) {
	var tests = []struct {
		before int
		after  int
	}{
		{
			before: 0,
			after:  272,
		},
		{
			before: 100,
			after:  272,
		},
		{
			before: 300,
			after:  300,
		},
		{
			before: 1024,
			after:  1024,
		},
	}

	for i, tt := range tests {
		p := make(Packet, tt.before)
		p.PadToMinSize()

		if want, got := tt.after, len(p); want != got {
			t.Fatalf("%02d: before %d, unexpected padded length: %d != %d",
				i, tt.before, want, got)
		}
	}
}

func TestRequestPacket(t *testing.T) {
	var tests = []struct {
		description string
		mt          MessageType
		chAddr      net.HardwareAddr
		cIAddr      net.IP
		xId         []byte
		broadcast   bool
		options     []Option
	}{
		{
			description: "discover request",
			mt:          Discover,
			chAddr:      net.HardwareAddr{1, 35, 69, 103, 117, 171}, // 01:23:45:67:89:ab
			cIAddr:      net.IP([]byte{192, 168, 1, 1}),
			xId:         []byte{0, 1, 2, 3},
			broadcast:   true,
			options:     nil,
		},
		{
			description: "request request",
			mt:          Request,
			chAddr:      net.HardwareAddr{222, 173, 190, 239, 222, 173}, // de:ad:be:ef:de:ad
			xId:         []byte{4, 5, 6, 7},
			broadcast:   false,
			options:     oneOptionSlice,
		},
		{
			description: "decline request",
			mt:          Decline,
			chAddr:      net.HardwareAddr{255, 255, 255, 255, 255, 255}, // ff:ff:ff:ff:ff:ff
			xId:         []byte{8, 9, 10, 11},
			broadcast:   true,
			options:     twoOptionsSlice,
		},
	}

	for i, tt := range tests {
		// Compare our basic test implementation's packet against the library's
		// implementation
		want := newRequestPacket(tt.mt, tt.chAddr, tt.cIAddr, tt.xId, tt.broadcast, tt.options)
		got := RequestPacket(tt.mt, tt.chAddr, tt.cIAddr, tt.xId, tt.broadcast, tt.options)

		if !bytes.Equal(want, got) {
			t.Fatalf("%02d: RequestPacket(), test %q, unexpected result: %v != %v",
				i, tt.description, want, got)
		}
	}
}

func TestReplyPacket(t *testing.T) {
	var tests = []struct {
		description   string
		mt            MessageType
		serverId      net.IP
		yIAddr        net.IP
		leaseDuration time.Duration
		options       []Option
	}{
		{
			description:   "offer reply",
			mt:            Offer,
			serverId:      []byte{192, 168, 1, 1},
			yIAddr:        []byte{192, 168, 1, 1},
			leaseDuration: 60 * time.Second,
			options:       nil,
		},
		{
			description:   "ACK reply",
			mt:            ACK,
			serverId:      []byte{10, 0, 0, 1},
			yIAddr:        []byte{192, 168, 1, 1},
			leaseDuration: 10 * time.Second,
			options:       oneOptionSlice,
		},
		{
			description:   "NAK reply",
			mt:            NAK,
			serverId:      []byte{8, 8, 8, 8},
			yIAddr:        []byte{8, 8, 4, 4},
			leaseDuration: 3600 * time.Second,
			options:       twoOptionsSlice,
		},
	}

	for i, tt := range tests {
		// Compare our basic test implementation's packet against the library's
		// implementation
		req := NewPacket(BootRequest)
		want := newReplyPacket(req, tt.mt, tt.serverId, tt.yIAddr, tt.leaseDuration, tt.options)
		got := ReplyPacket(req, tt.mt, tt.serverId, tt.yIAddr, tt.leaseDuration, tt.options)

		if !bytes.Equal(want, got) {
			t.Fatalf("%02d: ReplyPacket(), test %q, unexpected result: %v != %v",
				i, tt.description, want, got)
		}
	}
}

// newPacket mimics the raw logic of NewPacket, and verifies that its
// behavior does not change.
func newPacket(opCode OpCode) Packet {
	const ethernetHType = 1
	var cookie = []byte{99, 130, 83, 99}

	p := make(Packet, 241)
	p[0] = byte(opCode)
	p[1] = ethernetHType
	copy(p[236:240], cookie)
	p[240] = byte(End)

	return p
}

// newRequestPacket mimics the raw logic of RequestPacket, and verifies that
// its behavior does not change.
func newRequestPacket(mt MessageType, chAddr net.HardwareAddr, cIAddr net.IP, xId []byte, broadcast bool, options []Option) Packet {
	// Craft packet using our test method
	p := newPacket(BootRequest)

	// SetCHAddr
	copy(p[28:44], chAddr)
	p[2] = byte(len(chAddr))

	// SetXId
	copy(p[4:8], xId)

	// SetCIAddr
	if cIAddr != nil {
		copy(net.IP(p[12:16]), cIAddr.To4())
	}

	// SetBroadcast
	if broadcast {
		p[10:12][0] ^= 128
	}

	// AddOption already tested, so no need to duplicate the logic
	p.AddOption(OptionDHCPMessageType, []byte{byte(mt)})
	for _, o := range options {
		p.AddOption(o.Code, o.Value)
	}

	// PadToMinSize already tested
	p.PadToMinSize()

	return p
}

// newReplyPacket mimics the raw logic of ReplyPacket, and verifies that
// its behavior does not change.
func newReplyPacket(req Packet, mt MessageType, serverId, yIAddr net.IP, leaseDuration time.Duration, options []Option) Packet {
	// Craft packet using our test method
	p := newPacket(BootReply)

	// SetXId
	copy(p[4:8], req[4:8])

	// SetFlags
	copy(p[10:22], req[10:12])

	// SetYIAddr
	copy(p[16:20], yIAddr)

	// SetGIAddr
	copy(p[24:28], req[24:28])

	// SetCHAddr
	hLen := req[2]
	if hLen > 16 {
		hLen = 16
	}
	c := make([]byte, hLen)
	copy(c, p[28:28+hLen])

	copy(p[28:44], c)
	p[2] = byte(len(c))

	// SetSecs
	copy(p[8:10], req[8:10])

	// AddOption already tested, so no need to duplicate the logic
	p.AddOption(OptionDHCPMessageType, []byte{byte(mt)})
	p.AddOption(OptionServerIdentifier, []byte(serverId))
	p.AddOption(OptionIPAddressLeaseTime, OptionsLeaseTime(leaseDuration))
	for _, o := range options {
		p.AddOption(o.Code, o.Value)
	}

	// PadToMinSize already tested
	p.PadToMinSize()

	return p
}

// oneOptionSlice is a test helper of []Option with a single
// Option.
var oneOptionSlice = []Option{
	Option{
		Code:  OptionSubnetMask,
		Value: []byte{255, 255, 255, 0},
	},
}

// twoOptionSlice is a test helper of []Option with two
// Option values.
var twoOptionsSlice = []Option{
	Option{
		Code:  OptionSubnetMask,
		Value: []byte{255, 255, 255, 0},
	},
	Option{
		Code:  OptionDomainNameServer,
		Value: []byte{8, 8, 8, 8},
	},
}

// optionsTests are tests used when applying and stripping Options
// from Packets.
var optionsTests = []struct {
	description string
	options     []Option
}{
	{
		description: "nil options",
		options:     nil,
	},
	{
		description: "empty options",
		options:     []Option{},
	},
	{
		description: "padding option",
		options: []Option{
			Option{
				Code: Pad,
			},
		},
	},
	{
		description: "one option",
		options:     oneOptionSlice,
	},
	{
		description: "two options",
		options:     twoOptionsSlice,
	},
	{
		description: "four options",
		options: []Option{
			Option{
				Code:  OptionSubnetMask,
				Value: []byte{255, 255, 255, 0},
			},
			Option{
				Code:  OptionDomainNameServer,
				Value: []byte{8, 8, 8, 8},
			},
			Option{
				Code:  OptionTimeServer,
				Value: []byte{127, 0, 0, 1},
			},
			Option{
				Code:  OptionMessage,
				Value: []byte{'h', 'e', 'l', 'l', 'o', 'w', 'o', 'r', 'l', 'd'},
			},
		},
	},
}
