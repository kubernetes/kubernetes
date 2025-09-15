//+build !windows,!solaris

package dbus

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"net"
	"syscall"
)

type oobReader struct {
	conn *net.UnixConn
	oob  []byte
	buf  [4096]byte
}

func (o *oobReader) Read(b []byte) (n int, err error) {
	n, oobn, flags, _, err := o.conn.ReadMsgUnix(b, o.buf[:])
	if err != nil {
		return n, err
	}
	if flags&syscall.MSG_CTRUNC != 0 {
		return n, errors.New("dbus: control data truncated (too many fds received)")
	}
	o.oob = append(o.oob, o.buf[:oobn]...)
	return n, nil
}

type unixTransport struct {
	*net.UnixConn
	rdr        *oobReader
	hasUnixFDs bool
}

func newUnixTransport(keys string) (transport, error) {
	var err error

	t := new(unixTransport)
	abstract := getKey(keys, "abstract")
	path := getKey(keys, "path")
	switch {
	case abstract == "" && path == "":
		return nil, errors.New("dbus: invalid address (neither path nor abstract set)")
	case abstract != "" && path == "":
		t.UnixConn, err = net.DialUnix("unix", nil, &net.UnixAddr{Name: "@" + abstract, Net: "unix"})
		if err != nil {
			return nil, err
		}
		return t, nil
	case abstract == "" && path != "":
		t.UnixConn, err = net.DialUnix("unix", nil, &net.UnixAddr{Name: path, Net: "unix"})
		if err != nil {
			return nil, err
		}
		return t, nil
	default:
		return nil, errors.New("dbus: invalid address (both path and abstract set)")
	}
}

func init() {
	transports["unix"] = newUnixTransport
}

func (t *unixTransport) EnableUnixFDs() {
	t.hasUnixFDs = true
}

func (t *unixTransport) ReadMessage() (*Message, error) {
	var (
		blen, hlen uint32
		csheader   [16]byte
		headers    []header
		order      binary.ByteOrder
		unixfds    uint32
	)
	// To be sure that all bytes of out-of-band data are read, we use a special
	// reader that uses ReadUnix on the underlying connection instead of Read
	// and gathers the out-of-band data in a buffer.
	if t.rdr == nil {
		t.rdr = &oobReader{conn: t.UnixConn}
	} else {
		t.rdr.oob = nil
	}

	// read the first 16 bytes (the part of the header that has a constant size),
	// from which we can figure out the length of the rest of the message
	if _, err := io.ReadFull(t.rdr, csheader[:]); err != nil {
		return nil, err
	}
	switch csheader[0] {
	case 'l':
		order = binary.LittleEndian
	case 'B':
		order = binary.BigEndian
	default:
		return nil, InvalidMessageError("invalid byte order")
	}
	// csheader[4:8] -> length of message body, csheader[12:16] -> length of
	// header fields (without alignment)
	binary.Read(bytes.NewBuffer(csheader[4:8]), order, &blen)
	binary.Read(bytes.NewBuffer(csheader[12:]), order, &hlen)
	if hlen%8 != 0 {
		hlen += 8 - (hlen % 8)
	}

	// decode headers and look for unix fds
	headerdata := make([]byte, hlen+4)
	copy(headerdata, csheader[12:])
	if _, err := io.ReadFull(t.rdr, headerdata[4:]); err != nil {
		return nil, err
	}
	dec := newDecoder(bytes.NewBuffer(headerdata), order, make([]int, 0))
	dec.pos = 12
	vs, err := dec.Decode(Signature{"a(yv)"})
	if err != nil {
		return nil, err
	}
	Store(vs, &headers)
	for _, v := range headers {
		if v.Field == byte(FieldUnixFDs) {
			unixfds, _ = v.Variant.value.(uint32)
		}
	}
	all := make([]byte, 16+hlen+blen)
	copy(all, csheader[:])
	copy(all[16:], headerdata[4:])
	if _, err := io.ReadFull(t.rdr, all[16+hlen:]); err != nil {
		return nil, err
	}
	if unixfds != 0 {
		if !t.hasUnixFDs {
			return nil, errors.New("dbus: got unix fds on unsupported transport")
		}
		// read the fds from the OOB data
		scms, err := syscall.ParseSocketControlMessage(t.rdr.oob)
		if err != nil {
			return nil, err
		}
		if len(scms) != 1 {
			return nil, errors.New("dbus: received more than one socket control message")
		}
		fds, err := syscall.ParseUnixRights(&scms[0])
		if err != nil {
			return nil, err
		}
		msg, err := DecodeMessageWithFDs(bytes.NewBuffer(all), fds)
		if err != nil {
			return nil, err
		}
		// substitute the values in the message body (which are indices for the
		// array receiver via OOB) with the actual values
		for i, v := range msg.Body {
			switch index := v.(type) {
			case UnixFDIndex:
				if uint32(index) >= unixfds {
					return nil, InvalidMessageError("invalid index for unix fd")
				}
				msg.Body[i] = UnixFD(fds[index])
			case []UnixFDIndex:
				fdArray := make([]UnixFD, len(index))
				for k, j := range index {
					if uint32(j) >= unixfds {
						return nil, InvalidMessageError("invalid index for unix fd")
					}
					fdArray[k] = UnixFD(fds[j])
				}
				msg.Body[i] = fdArray
			}
		}
		return msg, nil
	}
	return DecodeMessage(bytes.NewBuffer(all))
}

func (t *unixTransport) SendMessage(msg *Message) error {
	fdcnt, err := msg.CountFds()
	if err != nil {
		return err
	}
	if fdcnt != 0 {
		if !t.hasUnixFDs {
			return errors.New("dbus: unix fd passing not enabled")
		}
		msg.Headers[FieldUnixFDs] = MakeVariant(uint32(fdcnt))
		buf := new(bytes.Buffer)
		fds, err := msg.EncodeToWithFDs(buf, nativeEndian)
		if err != nil {
			return err
		}
		oob := syscall.UnixRights(fds...)
		n, oobn, err := t.UnixConn.WriteMsgUnix(buf.Bytes(), oob, nil)
		if err != nil {
			return err
		}
		if n != buf.Len() || oobn != len(oob) {
			return io.ErrShortWrite
		}
	} else {
		if err := msg.EncodeTo(t, nativeEndian); err != nil {
			return err
		}
	}
	return nil
}

func (t *unixTransport) SupportsUnixFDs() bool {
	return true
}
