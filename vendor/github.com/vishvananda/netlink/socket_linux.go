package netlink

import (
	"errors"
	"fmt"
	"net"
	"syscall"

	"github.com/vishvananda/netlink/nl"
	"golang.org/x/sys/unix"
)

const (
	sizeofSocketID      = 0x30
	sizeofSocketRequest = sizeofSocketID + 0x8
	sizeofSocket        = sizeofSocketID + 0x18
)

type socketRequest struct {
	Family   uint8
	Protocol uint8
	Ext      uint8
	pad      uint8
	States   uint32
	ID       SocketID
}

type writeBuffer struct {
	Bytes []byte
	pos   int
}

func (b *writeBuffer) Write(c byte) {
	b.Bytes[b.pos] = c
	b.pos++
}

func (b *writeBuffer) Next(n int) []byte {
	s := b.Bytes[b.pos : b.pos+n]
	b.pos += n
	return s
}

func (r *socketRequest) Serialize() []byte {
	b := writeBuffer{Bytes: make([]byte, sizeofSocketRequest)}
	b.Write(r.Family)
	b.Write(r.Protocol)
	b.Write(r.Ext)
	b.Write(r.pad)
	native.PutUint32(b.Next(4), r.States)
	networkOrder.PutUint16(b.Next(2), r.ID.SourcePort)
	networkOrder.PutUint16(b.Next(2), r.ID.DestinationPort)
	if r.Family == unix.AF_INET6 {
		copy(b.Next(16), r.ID.Source)
		copy(b.Next(16), r.ID.Destination)
	} else {
		copy(b.Next(4), r.ID.Source.To4())
		b.Next(12)
		copy(b.Next(4), r.ID.Destination.To4())
		b.Next(12)
	}
	native.PutUint32(b.Next(4), r.ID.Interface)
	native.PutUint32(b.Next(4), r.ID.Cookie[0])
	native.PutUint32(b.Next(4), r.ID.Cookie[1])
	return b.Bytes
}

func (r *socketRequest) Len() int { return sizeofSocketRequest }

type readBuffer struct {
	Bytes []byte
	pos   int
}

func (b *readBuffer) Read() byte {
	c := b.Bytes[b.pos]
	b.pos++
	return c
}

func (b *readBuffer) Next(n int) []byte {
	s := b.Bytes[b.pos : b.pos+n]
	b.pos += n
	return s
}

func (s *Socket) deserialize(b []byte) error {
	if len(b) < sizeofSocket {
		return fmt.Errorf("socket data short read (%d); want %d", len(b), sizeofSocket)
	}
	rb := readBuffer{Bytes: b}
	s.Family = rb.Read()
	s.State = rb.Read()
	s.Timer = rb.Read()
	s.Retrans = rb.Read()
	s.ID.SourcePort = networkOrder.Uint16(rb.Next(2))
	s.ID.DestinationPort = networkOrder.Uint16(rb.Next(2))
	if s.Family == unix.AF_INET6 {
		s.ID.Source = net.IP(rb.Next(16))
		s.ID.Destination = net.IP(rb.Next(16))
	} else {
		s.ID.Source = net.IPv4(rb.Read(), rb.Read(), rb.Read(), rb.Read())
		rb.Next(12)
		s.ID.Destination = net.IPv4(rb.Read(), rb.Read(), rb.Read(), rb.Read())
		rb.Next(12)
	}
	s.ID.Interface = native.Uint32(rb.Next(4))
	s.ID.Cookie[0] = native.Uint32(rb.Next(4))
	s.ID.Cookie[1] = native.Uint32(rb.Next(4))
	s.Expires = native.Uint32(rb.Next(4))
	s.RQueue = native.Uint32(rb.Next(4))
	s.WQueue = native.Uint32(rb.Next(4))
	s.UID = native.Uint32(rb.Next(4))
	s.INode = native.Uint32(rb.Next(4))
	return nil
}

// SocketGet returns the Socket identified by its local and remote addresses.
func SocketGet(local, remote net.Addr) (*Socket, error) {
	localTCP, ok := local.(*net.TCPAddr)
	if !ok {
		return nil, ErrNotImplemented
	}
	remoteTCP, ok := remote.(*net.TCPAddr)
	if !ok {
		return nil, ErrNotImplemented
	}
	localIP := localTCP.IP.To4()
	if localIP == nil {
		return nil, ErrNotImplemented
	}
	remoteIP := remoteTCP.IP.To4()
	if remoteIP == nil {
		return nil, ErrNotImplemented
	}

	s, err := nl.Subscribe(unix.NETLINK_INET_DIAG)
	if err != nil {
		return nil, err
	}
	defer s.Close()
	req := nl.NewNetlinkRequest(nl.SOCK_DIAG_BY_FAMILY, 0)
	req.AddData(&socketRequest{
		Family:   unix.AF_INET,
		Protocol: unix.IPPROTO_TCP,
		ID: SocketID{
			SourcePort:      uint16(localTCP.Port),
			DestinationPort: uint16(remoteTCP.Port),
			Source:          localIP,
			Destination:     remoteIP,
			Cookie:          [2]uint32{nl.TCPDIAG_NOCOOKIE, nl.TCPDIAG_NOCOOKIE},
		},
	})
	s.Send(req)
	msgs, from, err := s.Receive()
	if err != nil {
		return nil, err
	}
	if from.Pid != nl.PidKernel {
		return nil, fmt.Errorf("Wrong sender portid %d, expected %d", from.Pid, nl.PidKernel)
	}
	if len(msgs) == 0 {
		return nil, errors.New("no message nor error from netlink")
	}
	if len(msgs) > 2 {
		return nil, fmt.Errorf("multiple (%d) matching sockets", len(msgs))
	}
	sock := &Socket{}
	if err := sock.deserialize(msgs[0].Data); err != nil {
		return nil, err
	}
	return sock, nil
}

// SocketDiagTCPInfo requests INET_DIAG_INFO for TCP protocol for specified family type.
func SocketDiagTCPInfo(family uint8) ([]*InetDiagTCPInfoResp, error) {
	s, err := nl.Subscribe(unix.NETLINK_INET_DIAG)
	if err != nil {
		return nil, err
	}
	defer s.Close()

	req := nl.NewNetlinkRequest(nl.SOCK_DIAG_BY_FAMILY, unix.NLM_F_DUMP)
	req.AddData(&socketRequest{
		Family:   family,
		Protocol: unix.IPPROTO_TCP,
		Ext:      INET_DIAG_INFO,
		States:   uint32(0xfff), // All TCP states
	})
	s.Send(req)

	var result []*InetDiagTCPInfoResp
loop:
	for {
		msgs, from, err := s.Receive()
		if err != nil {
			return nil, err
		}
		if from.Pid != nl.PidKernel {
			return nil, fmt.Errorf("Wrong sender portid %d, expected %d", from.Pid, nl.PidKernel)
		}
		if len(msgs) == 0 {
			return nil, errors.New("no message nor error from netlink")
		}

		for _, m := range msgs {
			switch m.Header.Type {
			case unix.NLMSG_DONE:
				break loop
			case unix.NLMSG_ERROR:
				native := nl.NativeEndian()
				error := int32(native.Uint32(m.Data[0:4]))
				return nil, syscall.Errno(-error)
			}
			sockInfo := &Socket{}
			if err := sockInfo.deserialize(m.Data); err != nil {
				return nil, err
			}
			attrs, err := nl.ParseRouteAttr(m.Data[sizeofSocket:])
			if err != nil {
				return nil, err
			}
			var tcpInfo *TCPInfo
			for _, a := range attrs {
				if a.Attr.Type == INET_DIAG_INFO {
					tcpInfo = &TCPInfo{}
					if err := tcpInfo.deserialize(a.Value); err != nil {
						return nil, err
					}
					break
				}
			}
			r := &InetDiagTCPInfoResp{InetDiagMsg: sockInfo, TCPInfo: tcpInfo}
			result = append(result, r)
		}
	}
	return result, nil
}
