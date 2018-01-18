package dhcp4

import (
	"net"
	"time"
)

// A DHCP packet
type Packet []byte

func (p Packet) OpCode() OpCode { return OpCode(p[0]) }
func (p Packet) HType() byte    { return p[1] }
func (p Packet) HLen() byte     { return p[2] }
func (p Packet) Hops() byte     { return p[3] }
func (p Packet) XId() []byte    { return p[4:8] }
func (p Packet) Secs() []byte   { return p[8:10] } // Never Used?
func (p Packet) Flags() []byte  { return p[10:12] }
func (p Packet) CIAddr() net.IP { return net.IP(p[12:16]) }
func (p Packet) YIAddr() net.IP { return net.IP(p[16:20]) }
func (p Packet) SIAddr() net.IP { return net.IP(p[20:24]) }
func (p Packet) GIAddr() net.IP { return net.IP(p[24:28]) }
func (p Packet) CHAddr() net.HardwareAddr {
	hLen := p.HLen()
	if hLen > 16 { // Prevent chaddr exceeding p boundary
		hLen = 16
	}
	return net.HardwareAddr(p[28 : 28+hLen]) // max endPos 44
}

// 192 bytes of zeros BOOTP legacy
func (p Packet) Cookie() []byte { return p[236:240] }
func (p Packet) Options() []byte {
	if len(p) > 240 {
		return p[240:]
	}
	return nil
}

func (p Packet) Broadcast() bool { return p.Flags()[0] > 127 }

func (p Packet) SetBroadcast(broadcast bool) {
	if p.Broadcast() != broadcast {
		p.Flags()[0] ^= 128
	}
}

func (p Packet) SetOpCode(c OpCode) { p[0] = byte(c) }
func (p Packet) SetCHAddr(a net.HardwareAddr) {
	copy(p[28:44], a)
	p[2] = byte(len(a))
}
func (p Packet) SetHType(hType byte)     { p[1] = hType }
func (p Packet) SetCookie(cookie []byte) { copy(p.Cookie(), cookie) }
func (p Packet) SetHops(hops byte)       { p[3] = hops }
func (p Packet) SetXId(xId []byte)       { copy(p.XId(), xId) }
func (p Packet) SetSecs(secs []byte)     { copy(p.Secs(), secs) }
func (p Packet) SetFlags(flags []byte)   { copy(p.Flags(), flags) }
func (p Packet) SetCIAddr(ip net.IP)     { copy(p.CIAddr(), ip.To4()) }
func (p Packet) SetYIAddr(ip net.IP)     { copy(p.YIAddr(), ip.To4()) }
func (p Packet) SetSIAddr(ip net.IP)     { copy(p.SIAddr(), ip.To4()) }
func (p Packet) SetGIAddr(ip net.IP)     { copy(p.GIAddr(), ip.To4()) }

// Parses the packet's options into an Options map
func (p Packet) ParseOptions() Options {
	opts := p.Options()
	options := make(Options, 10)
	for len(opts) >= 2 && OptionCode(opts[0]) != End {
		if OptionCode(opts[0]) == Pad {
			opts = opts[1:]
			continue
		}
		size := int(opts[1])
		if len(opts) < 2+size {
			break
		}
		options[OptionCode(opts[0])] = opts[2 : 2+size]
		opts = opts[2+size:]
	}
	return options
}

func NewPacket(opCode OpCode) Packet {
	p := make(Packet, 241)
	p.SetOpCode(opCode)
	p.SetHType(1) // Ethernet
	p.SetCookie([]byte{99, 130, 83, 99})
	p[240] = byte(End)
	return p
}

// Appends a DHCP option to the end of a packet
func (p *Packet) AddOption(o OptionCode, value []byte) {
	*p = append((*p)[:len(*p)-1], []byte{byte(o), byte(len(value))}...) // Strip off End, Add OptionCode and Length
	*p = append(*p, value...)                                           // Add Option Value
	*p = append(*p, byte(End))                                          // Add on new End
}

// Removes all options from packet.
func (p *Packet) StripOptions() {
	*p = append((*p)[:240], byte(End))
}

// Creates a request packet that a Client would send to a server.
func RequestPacket(mt MessageType, chAddr net.HardwareAddr, cIAddr net.IP, xId []byte, broadcast bool, options []Option) Packet {
	p := NewPacket(BootRequest)
	p.SetCHAddr(chAddr)
	p.SetXId(xId)
	if cIAddr != nil {
		p.SetCIAddr(cIAddr)
	}
	p.SetBroadcast(broadcast)
	p.AddOption(OptionDHCPMessageType, []byte{byte(mt)})
	for _, o := range options {
		p.AddOption(o.Code, o.Value)
	}
	p.PadToMinSize()
	return p
}

// ReplyPacket creates a reply packet that a Server would send to a client.
// It uses the req Packet param to copy across common/necessary fields to
// associate the reply the request.
func ReplyPacket(req Packet, mt MessageType, serverId, yIAddr net.IP, leaseDuration time.Duration, options []Option) Packet {
	p := NewPacket(BootReply)
	p.SetXId(req.XId())
	p.SetFlags(req.Flags())
	p.SetYIAddr(yIAddr)
	p.SetGIAddr(req.GIAddr())
	p.SetCHAddr(req.CHAddr())
	p.AddOption(OptionDHCPMessageType, []byte{byte(mt)})
	p.AddOption(OptionServerIdentifier, []byte(serverId))
	if leaseDuration > 0 {
		p.AddOption(OptionIPAddressLeaseTime, OptionsLeaseTime(leaseDuration))
	}
	for _, o := range options {
		p.AddOption(o.Code, o.Value)
	}
	p.PadToMinSize()
	return p
}

// PadToMinSize pads a packet so that when sent over UDP, the entire packet,
// is 300 bytes (BOOTP min), to be compatible with really old devices.
var padder [272]byte

func (p *Packet) PadToMinSize() {
	if n := len(*p); n < 272 {
		*p = append(*p, padder[:272-n]...)
	}
}
