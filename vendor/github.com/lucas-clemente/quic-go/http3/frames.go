package http3

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/quicvarint"
)

type frame interface{}

func parseNextFrame(r io.Reader) (frame, error) {
	qr := quicvarint.NewReader(r)
	t, err := quicvarint.Read(qr)
	if err != nil {
		return nil, err
	}
	l, err := quicvarint.Read(qr)
	if err != nil {
		return nil, err
	}

	switch t {
	case 0x0:
		return &dataFrame{Length: l}, nil
	case 0x1:
		return &headersFrame{Length: l}, nil
	case 0x4:
		return parseSettingsFrame(r, l)
	case 0x3: // CANCEL_PUSH
		fallthrough
	case 0x5: // PUSH_PROMISE
		fallthrough
	case 0x7: // GOAWAY
		fallthrough
	case 0xd: // MAX_PUSH_ID
		fallthrough
	case 0xe: // DUPLICATE_PUSH
		fallthrough
	default:
		// skip over unknown frames
		if _, err := io.CopyN(ioutil.Discard, qr, int64(l)); err != nil {
			return nil, err
		}
		return parseNextFrame(qr)
	}
}

type dataFrame struct {
	Length uint64
}

func (f *dataFrame) Write(b *bytes.Buffer) {
	quicvarint.Write(b, 0x0)
	quicvarint.Write(b, f.Length)
}

type headersFrame struct {
	Length uint64
}

func (f *headersFrame) Write(b *bytes.Buffer) {
	quicvarint.Write(b, 0x1)
	quicvarint.Write(b, f.Length)
}

const settingDatagram = 0x276

type settingsFrame struct {
	Datagram bool
	other    map[uint64]uint64 // all settings that we don't explicitly recognize
}

func parseSettingsFrame(r io.Reader, l uint64) (*settingsFrame, error) {
	if l > 8*(1<<10) {
		return nil, fmt.Errorf("unexpected size for SETTINGS frame: %d", l)
	}
	buf := make([]byte, l)
	if _, err := io.ReadFull(r, buf); err != nil {
		if err == io.ErrUnexpectedEOF {
			return nil, io.EOF
		}
		return nil, err
	}
	frame := &settingsFrame{}
	b := bytes.NewReader(buf)
	var readDatagram bool
	for b.Len() > 0 {
		id, err := quicvarint.Read(b)
		if err != nil { // should not happen. We allocated the whole frame already.
			return nil, err
		}
		val, err := quicvarint.Read(b)
		if err != nil { // should not happen. We allocated the whole frame already.
			return nil, err
		}

		switch id {
		case settingDatagram:
			if readDatagram {
				return nil, fmt.Errorf("duplicate setting: %d", id)
			}
			readDatagram = true
			if val != 0 && val != 1 {
				return nil, fmt.Errorf("invalid value for H3_DATAGRAM: %d", val)
			}
			frame.Datagram = val == 1
		default:
			if _, ok := frame.other[id]; ok {
				return nil, fmt.Errorf("duplicate setting: %d", id)
			}
			if frame.other == nil {
				frame.other = make(map[uint64]uint64)
			}
			frame.other[id] = val
		}
	}
	return frame, nil
}

func (f *settingsFrame) Write(b *bytes.Buffer) {
	quicvarint.Write(b, 0x4)
	var l protocol.ByteCount
	for id, val := range f.other {
		l += quicvarint.Len(id) + quicvarint.Len(val)
	}
	if f.Datagram {
		l += quicvarint.Len(settingDatagram) + quicvarint.Len(1)
	}
	quicvarint.Write(b, uint64(l))
	if f.Datagram {
		quicvarint.Write(b, settingDatagram)
		quicvarint.Write(b, 1)
	}
	for id, val := range f.other {
		quicvarint.Write(b, id)
		quicvarint.Write(b, val)
	}
}
