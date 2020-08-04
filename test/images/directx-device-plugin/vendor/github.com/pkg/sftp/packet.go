package sftp

import (
	"bytes"
	"encoding"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"reflect"

	"github.com/pkg/errors"
)

var (
	errShortPacket           = errors.New("packet too short")
	errUnknownExtendedPacket = errors.New("unknown extended packet")
)

const (
	debugDumpTxPacket      = false
	debugDumpRxPacket      = false
	debugDumpTxPacketBytes = false
	debugDumpRxPacketBytes = false
)

func marshalUint32(b []byte, v uint32) []byte {
	return append(b, byte(v>>24), byte(v>>16), byte(v>>8), byte(v))
}

func marshalUint64(b []byte, v uint64) []byte {
	return marshalUint32(marshalUint32(b, uint32(v>>32)), uint32(v))
}

func marshalString(b []byte, v string) []byte {
	return append(marshalUint32(b, uint32(len(v))), v...)
}

func marshal(b []byte, v interface{}) []byte {
	if v == nil {
		return b
	}
	switch v := v.(type) {
	case uint8:
		return append(b, v)
	case uint32:
		return marshalUint32(b, v)
	case uint64:
		return marshalUint64(b, v)
	case string:
		return marshalString(b, v)
	case os.FileInfo:
		return marshalFileInfo(b, v)
	default:
		switch d := reflect.ValueOf(v); d.Kind() {
		case reflect.Struct:
			for i, n := 0, d.NumField(); i < n; i++ {
				b = append(marshal(b, d.Field(i).Interface()))
			}
			return b
		case reflect.Slice:
			for i, n := 0, d.Len(); i < n; i++ {
				b = append(marshal(b, d.Index(i).Interface()))
			}
			return b
		default:
			panic(fmt.Sprintf("marshal(%#v): cannot handle type %T", v, v))
		}
	}
}

func unmarshalUint32(b []byte) (uint32, []byte) {
	v := uint32(b[3]) | uint32(b[2])<<8 | uint32(b[1])<<16 | uint32(b[0])<<24
	return v, b[4:]
}

func unmarshalUint32Safe(b []byte) (uint32, []byte, error) {
	var v uint32
	if len(b) < 4 {
		return 0, nil, errShortPacket
	}
	v, b = unmarshalUint32(b)
	return v, b, nil
}

func unmarshalUint64(b []byte) (uint64, []byte) {
	h, b := unmarshalUint32(b)
	l, b := unmarshalUint32(b)
	return uint64(h)<<32 | uint64(l), b
}

func unmarshalUint64Safe(b []byte) (uint64, []byte, error) {
	var v uint64
	if len(b) < 8 {
		return 0, nil, errShortPacket
	}
	v, b = unmarshalUint64(b)
	return v, b, nil
}

func unmarshalString(b []byte) (string, []byte) {
	n, b := unmarshalUint32(b)
	return string(b[:n]), b[n:]
}

func unmarshalStringSafe(b []byte) (string, []byte, error) {
	n, b, err := unmarshalUint32Safe(b)
	if err != nil {
		return "", nil, err
	}
	if int64(n) > int64(len(b)) {
		return "", nil, errShortPacket
	}
	return string(b[:n]), b[n:], nil
}

// sendPacket marshals p according to RFC 4234.
func sendPacket(w io.Writer, m encoding.BinaryMarshaler) error {
	bb, err := m.MarshalBinary()
	if err != nil {
		return errors.Errorf("binary marshaller failed: %v", err)
	}
	if debugDumpTxPacketBytes {
		debug("send packet: %s %d bytes %x", fxp(bb[0]), len(bb), bb[1:])
	} else if debugDumpTxPacket {
		debug("send packet: %s %d bytes", fxp(bb[0]), len(bb))
	}
	l := uint32(len(bb))
	hdr := []byte{byte(l >> 24), byte(l >> 16), byte(l >> 8), byte(l)}
	_, err = w.Write(hdr)
	if err != nil {
		return errors.Errorf("failed to send packet header: %v", err)
	}
	_, err = w.Write(bb)
	if err != nil {
		return errors.Errorf("failed to send packet body: %v", err)
	}
	return nil
}

func recvPacket(r io.Reader) (uint8, []byte, error) {
	var b = []byte{0, 0, 0, 0}
	if _, err := io.ReadFull(r, b); err != nil {
		return 0, nil, err
	}
	l, _ := unmarshalUint32(b)
	b = make([]byte, l)
	if _, err := io.ReadFull(r, b); err != nil {
		debug("recv packet %d bytes: err %v", l, err)
		return 0, nil, err
	}
	if debugDumpRxPacketBytes {
		debug("recv packet: %s %d bytes %x", fxp(b[0]), l, b[1:])
	} else if debugDumpRxPacket {
		debug("recv packet: %s %d bytes", fxp(b[0]), l)
	}
	return b[0], b[1:], nil
}

type extensionPair struct {
	Name string
	Data string
}

func unmarshalExtensionPair(b []byte) (extensionPair, []byte, error) {
	var ep extensionPair
	var err error
	ep.Name, b, err = unmarshalStringSafe(b)
	if err != nil {
		return ep, b, err
	}
	ep.Data, b, err = unmarshalStringSafe(b)
	if err != nil {
		return ep, b, err
	}
	return ep, b, err
}

// Here starts the definition of packets along with their MarshalBinary
// implementations.
// Manually writing the marshalling logic wins us a lot of time and
// allocation.

type sshFxInitPacket struct {
	Version    uint32
	Extensions []extensionPair
}

func (p sshFxInitPacket) MarshalBinary() ([]byte, error) {
	l := 1 + 4 // byte + uint32
	for _, e := range p.Extensions {
		l += 4 + len(e.Name) + 4 + len(e.Data)
	}

	b := make([]byte, 0, l)
	b = append(b, ssh_FXP_INIT)
	b = marshalUint32(b, p.Version)
	for _, e := range p.Extensions {
		b = marshalString(b, e.Name)
		b = marshalString(b, e.Data)
	}
	return b, nil
}

func (p *sshFxInitPacket) UnmarshalBinary(b []byte) error {
	var err error
	if p.Version, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	}
	for len(b) > 0 {
		var ep extensionPair
		ep, b, err = unmarshalExtensionPair(b)
		if err != nil {
			return err
		}
		p.Extensions = append(p.Extensions, ep)
	}
	return nil
}

type sshFxVersionPacket struct {
	Version    uint32
	Extensions []struct {
		Name, Data string
	}
}

func (p sshFxVersionPacket) MarshalBinary() ([]byte, error) {
	l := 1 + 4 // byte + uint32
	for _, e := range p.Extensions {
		l += 4 + len(e.Name) + 4 + len(e.Data)
	}

	b := make([]byte, 0, l)
	b = append(b, ssh_FXP_VERSION)
	b = marshalUint32(b, p.Version)
	for _, e := range p.Extensions {
		b = marshalString(b, e.Name)
		b = marshalString(b, e.Data)
	}
	return b, nil
}

func marshalIDString(packetType byte, id uint32, str string) ([]byte, error) {
	l := 1 + 4 + // type(byte) + uint32
		4 + len(str)

	b := make([]byte, 0, l)
	b = append(b, packetType)
	b = marshalUint32(b, id)
	b = marshalString(b, str)
	return b, nil
}

func unmarshalIDString(b []byte, id *uint32, str *string) error {
	var err error
	*id, b, err = unmarshalUint32Safe(b)
	if err != nil {
		return err
	}
	*str, b, err = unmarshalStringSafe(b)
	return err
}

type sshFxpReaddirPacket struct {
	ID     uint32
	Handle string
}

func (p sshFxpReaddirPacket) id() uint32 { return p.ID }

func (p sshFxpReaddirPacket) MarshalBinary() ([]byte, error) {
	return marshalIDString(ssh_FXP_READDIR, p.ID, p.Handle)
}

func (p *sshFxpReaddirPacket) UnmarshalBinary(b []byte) error {
	return unmarshalIDString(b, &p.ID, &p.Handle)
}

type sshFxpOpendirPacket struct {
	ID   uint32
	Path string
}

func (p sshFxpOpendirPacket) id() uint32 { return p.ID }

func (p sshFxpOpendirPacket) MarshalBinary() ([]byte, error) {
	return marshalIDString(ssh_FXP_OPENDIR, p.ID, p.Path)
}

func (p *sshFxpOpendirPacket) UnmarshalBinary(b []byte) error {
	return unmarshalIDString(b, &p.ID, &p.Path)
}

type sshFxpLstatPacket struct {
	ID   uint32
	Path string
}

func (p sshFxpLstatPacket) id() uint32 { return p.ID }

func (p sshFxpLstatPacket) MarshalBinary() ([]byte, error) {
	return marshalIDString(ssh_FXP_LSTAT, p.ID, p.Path)
}

func (p *sshFxpLstatPacket) UnmarshalBinary(b []byte) error {
	return unmarshalIDString(b, &p.ID, &p.Path)
}

type sshFxpStatPacket struct {
	ID   uint32
	Path string
}

func (p sshFxpStatPacket) id() uint32 { return p.ID }

func (p sshFxpStatPacket) MarshalBinary() ([]byte, error) {
	return marshalIDString(ssh_FXP_STAT, p.ID, p.Path)
}

func (p *sshFxpStatPacket) UnmarshalBinary(b []byte) error {
	return unmarshalIDString(b, &p.ID, &p.Path)
}

type sshFxpFstatPacket struct {
	ID     uint32
	Handle string
}

func (p sshFxpFstatPacket) id() uint32 { return p.ID }

func (p sshFxpFstatPacket) MarshalBinary() ([]byte, error) {
	return marshalIDString(ssh_FXP_FSTAT, p.ID, p.Handle)
}

func (p *sshFxpFstatPacket) UnmarshalBinary(b []byte) error {
	return unmarshalIDString(b, &p.ID, &p.Handle)
}

type sshFxpClosePacket struct {
	ID     uint32
	Handle string
}

func (p sshFxpClosePacket) id() uint32 { return p.ID }

func (p sshFxpClosePacket) MarshalBinary() ([]byte, error) {
	return marshalIDString(ssh_FXP_CLOSE, p.ID, p.Handle)
}

func (p *sshFxpClosePacket) UnmarshalBinary(b []byte) error {
	return unmarshalIDString(b, &p.ID, &p.Handle)
}

type sshFxpRemovePacket struct {
	ID       uint32
	Filename string
}

func (p sshFxpRemovePacket) id() uint32 { return p.ID }

func (p sshFxpRemovePacket) MarshalBinary() ([]byte, error) {
	return marshalIDString(ssh_FXP_REMOVE, p.ID, p.Filename)
}

func (p *sshFxpRemovePacket) UnmarshalBinary(b []byte) error {
	return unmarshalIDString(b, &p.ID, &p.Filename)
}

type sshFxpRmdirPacket struct {
	ID   uint32
	Path string
}

func (p sshFxpRmdirPacket) id() uint32 { return p.ID }

func (p sshFxpRmdirPacket) MarshalBinary() ([]byte, error) {
	return marshalIDString(ssh_FXP_RMDIR, p.ID, p.Path)
}

func (p *sshFxpRmdirPacket) UnmarshalBinary(b []byte) error {
	return unmarshalIDString(b, &p.ID, &p.Path)
}

type sshFxpSymlinkPacket struct {
	ID         uint32
	Targetpath string
	Linkpath   string
}

func (p sshFxpSymlinkPacket) id() uint32 { return p.ID }

func (p sshFxpSymlinkPacket) MarshalBinary() ([]byte, error) {
	l := 1 + 4 + // type(byte) + uint32
		4 + len(p.Targetpath) +
		4 + len(p.Linkpath)

	b := make([]byte, 0, l)
	b = append(b, ssh_FXP_SYMLINK)
	b = marshalUint32(b, p.ID)
	b = marshalString(b, p.Targetpath)
	b = marshalString(b, p.Linkpath)
	return b, nil
}

func (p *sshFxpSymlinkPacket) UnmarshalBinary(b []byte) error {
	var err error
	if p.ID, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	} else if p.Targetpath, b, err = unmarshalStringSafe(b); err != nil {
		return err
	} else if p.Linkpath, b, err = unmarshalStringSafe(b); err != nil {
		return err
	}
	return nil
}

type sshFxpReadlinkPacket struct {
	ID   uint32
	Path string
}

func (p sshFxpReadlinkPacket) id() uint32 { return p.ID }

func (p sshFxpReadlinkPacket) MarshalBinary() ([]byte, error) {
	return marshalIDString(ssh_FXP_READLINK, p.ID, p.Path)
}

func (p *sshFxpReadlinkPacket) UnmarshalBinary(b []byte) error {
	return unmarshalIDString(b, &p.ID, &p.Path)
}

type sshFxpRealpathPacket struct {
	ID   uint32
	Path string
}

func (p sshFxpRealpathPacket) id() uint32 { return p.ID }

func (p sshFxpRealpathPacket) MarshalBinary() ([]byte, error) {
	return marshalIDString(ssh_FXP_REALPATH, p.ID, p.Path)
}

func (p *sshFxpRealpathPacket) UnmarshalBinary(b []byte) error {
	return unmarshalIDString(b, &p.ID, &p.Path)
}

type sshFxpNameAttr struct {
	Name     string
	LongName string
	Attrs    []interface{}
}

func (p sshFxpNameAttr) MarshalBinary() ([]byte, error) {
	b := []byte{}
	b = marshalString(b, p.Name)
	b = marshalString(b, p.LongName)
	for _, attr := range p.Attrs {
		b = marshal(b, attr)
	}
	return b, nil
}

type sshFxpNamePacket struct {
	ID        uint32
	NameAttrs []sshFxpNameAttr
}

func (p sshFxpNamePacket) MarshalBinary() ([]byte, error) {
	b := []byte{}
	b = append(b, ssh_FXP_NAME)
	b = marshalUint32(b, p.ID)
	b = marshalUint32(b, uint32(len(p.NameAttrs)))
	for _, na := range p.NameAttrs {
		ab, err := na.MarshalBinary()
		if err != nil {
			return nil, err
		}

		b = append(b, ab...)
	}
	return b, nil
}

type sshFxpOpenPacket struct {
	ID     uint32
	Path   string
	Pflags uint32
	Flags  uint32 // ignored
}

func (p sshFxpOpenPacket) id() uint32 { return p.ID }

func (p sshFxpOpenPacket) MarshalBinary() ([]byte, error) {
	l := 1 + 4 +
		4 + len(p.Path) +
		4 + 4

	b := make([]byte, 0, l)
	b = append(b, ssh_FXP_OPEN)
	b = marshalUint32(b, p.ID)
	b = marshalString(b, p.Path)
	b = marshalUint32(b, p.Pflags)
	b = marshalUint32(b, p.Flags)
	return b, nil
}

func (p *sshFxpOpenPacket) UnmarshalBinary(b []byte) error {
	var err error
	if p.ID, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	} else if p.Path, b, err = unmarshalStringSafe(b); err != nil {
		return err
	} else if p.Pflags, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	} else if p.Flags, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	}
	return nil
}

type sshFxpReadPacket struct {
	ID     uint32
	Handle string
	Offset uint64
	Len    uint32
}

func (p sshFxpReadPacket) id() uint32 { return p.ID }

func (p sshFxpReadPacket) MarshalBinary() ([]byte, error) {
	l := 1 + 4 + // type(byte) + uint32
		4 + len(p.Handle) +
		8 + 4 // uint64 + uint32

	b := make([]byte, 0, l)
	b = append(b, ssh_FXP_READ)
	b = marshalUint32(b, p.ID)
	b = marshalString(b, p.Handle)
	b = marshalUint64(b, p.Offset)
	b = marshalUint32(b, p.Len)
	return b, nil
}

func (p *sshFxpReadPacket) UnmarshalBinary(b []byte) error {
	var err error
	if p.ID, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	} else if p.Handle, b, err = unmarshalStringSafe(b); err != nil {
		return err
	} else if p.Offset, b, err = unmarshalUint64Safe(b); err != nil {
		return err
	} else if p.Len, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	}
	return nil
}

type sshFxpRenamePacket struct {
	ID      uint32
	Oldpath string
	Newpath string
}

func (p sshFxpRenamePacket) id() uint32 { return p.ID }

func (p sshFxpRenamePacket) MarshalBinary() ([]byte, error) {
	l := 1 + 4 + // type(byte) + uint32
		4 + len(p.Oldpath) +
		4 + len(p.Newpath)

	b := make([]byte, 0, l)
	b = append(b, ssh_FXP_RENAME)
	b = marshalUint32(b, p.ID)
	b = marshalString(b, p.Oldpath)
	b = marshalString(b, p.Newpath)
	return b, nil
}

func (p *sshFxpRenamePacket) UnmarshalBinary(b []byte) error {
	var err error
	if p.ID, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	} else if p.Oldpath, b, err = unmarshalStringSafe(b); err != nil {
		return err
	} else if p.Newpath, b, err = unmarshalStringSafe(b); err != nil {
		return err
	}
	return nil
}

type sshFxpWritePacket struct {
	ID     uint32
	Handle string
	Offset uint64
	Length uint32
	Data   []byte
}

func (p sshFxpWritePacket) id() uint32 { return p.ID }

func (p sshFxpWritePacket) MarshalBinary() ([]byte, error) {
	l := 1 + 4 + // type(byte) + uint32
		4 + len(p.Handle) +
		8 + 4 + // uint64 + uint32
		len(p.Data)

	b := make([]byte, 0, l)
	b = append(b, ssh_FXP_WRITE)
	b = marshalUint32(b, p.ID)
	b = marshalString(b, p.Handle)
	b = marshalUint64(b, p.Offset)
	b = marshalUint32(b, p.Length)
	b = append(b, p.Data...)
	return b, nil
}

func (p *sshFxpWritePacket) UnmarshalBinary(b []byte) error {
	var err error
	if p.ID, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	} else if p.Handle, b, err = unmarshalStringSafe(b); err != nil {
		return err
	} else if p.Offset, b, err = unmarshalUint64Safe(b); err != nil {
		return err
	} else if p.Length, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	} else if uint32(len(b)) < p.Length {
		return errShortPacket
	}

	p.Data = append([]byte{}, b[:p.Length]...)
	return nil
}

type sshFxpMkdirPacket struct {
	ID    uint32
	Path  string
	Flags uint32 // ignored
}

func (p sshFxpMkdirPacket) id() uint32 { return p.ID }

func (p sshFxpMkdirPacket) MarshalBinary() ([]byte, error) {
	l := 1 + 4 + // type(byte) + uint32
		4 + len(p.Path) +
		4 // uint32

	b := make([]byte, 0, l)
	b = append(b, ssh_FXP_MKDIR)
	b = marshalUint32(b, p.ID)
	b = marshalString(b, p.Path)
	b = marshalUint32(b, p.Flags)
	return b, nil
}

func (p *sshFxpMkdirPacket) UnmarshalBinary(b []byte) error {
	var err error
	if p.ID, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	} else if p.Path, b, err = unmarshalStringSafe(b); err != nil {
		return err
	} else if p.Flags, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	}
	return nil
}

type sshFxpSetstatPacket struct {
	ID    uint32
	Path  string
	Flags uint32
	Attrs interface{}
}

type sshFxpFsetstatPacket struct {
	ID     uint32
	Handle string
	Flags  uint32
	Attrs  interface{}
}

func (p sshFxpSetstatPacket) id() uint32  { return p.ID }
func (p sshFxpFsetstatPacket) id() uint32 { return p.ID }

func (p sshFxpSetstatPacket) MarshalBinary() ([]byte, error) {
	l := 1 + 4 + // type(byte) + uint32
		4 + len(p.Path) +
		4 // uint32 + uint64

	b := make([]byte, 0, l)
	b = append(b, ssh_FXP_SETSTAT)
	b = marshalUint32(b, p.ID)
	b = marshalString(b, p.Path)
	b = marshalUint32(b, p.Flags)
	b = marshal(b, p.Attrs)
	return b, nil
}

func (p sshFxpFsetstatPacket) MarshalBinary() ([]byte, error) {
	l := 1 + 4 + // type(byte) + uint32
		4 + len(p.Handle) +
		4 // uint32 + uint64

	b := make([]byte, 0, l)
	b = append(b, ssh_FXP_FSETSTAT)
	b = marshalUint32(b, p.ID)
	b = marshalString(b, p.Handle)
	b = marshalUint32(b, p.Flags)
	b = marshal(b, p.Attrs)
	return b, nil
}

func (p *sshFxpSetstatPacket) UnmarshalBinary(b []byte) error {
	var err error
	if p.ID, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	} else if p.Path, b, err = unmarshalStringSafe(b); err != nil {
		return err
	} else if p.Flags, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	}
	p.Attrs = b
	return nil
}

func (p *sshFxpFsetstatPacket) UnmarshalBinary(b []byte) error {
	var err error
	if p.ID, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	} else if p.Handle, b, err = unmarshalStringSafe(b); err != nil {
		return err
	} else if p.Flags, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	}
	p.Attrs = b
	return nil
}

type sshFxpHandlePacket struct {
	ID     uint32
	Handle string
}

func (p sshFxpHandlePacket) MarshalBinary() ([]byte, error) {
	b := []byte{ssh_FXP_HANDLE}
	b = marshalUint32(b, p.ID)
	b = marshalString(b, p.Handle)
	return b, nil
}

type sshFxpStatusPacket struct {
	ID uint32
	StatusError
}

func (p sshFxpStatusPacket) MarshalBinary() ([]byte, error) {
	b := []byte{ssh_FXP_STATUS}
	b = marshalUint32(b, p.ID)
	b = marshalStatus(b, p.StatusError)
	return b, nil
}

type sshFxpDataPacket struct {
	ID     uint32
	Length uint32
	Data   []byte
}

func (p sshFxpDataPacket) MarshalBinary() ([]byte, error) {
	b := []byte{ssh_FXP_DATA}
	b = marshalUint32(b, p.ID)
	b = marshalUint32(b, p.Length)
	b = append(b, p.Data[:p.Length]...)
	return b, nil
}

func (p *sshFxpDataPacket) UnmarshalBinary(b []byte) error {
	var err error
	if p.ID, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	} else if p.Length, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	} else if uint32(len(b)) < p.Length {
		return errors.New("truncated packet")
	}

	p.Data = make([]byte, p.Length)
	copy(p.Data, b)
	return nil
}

type sshFxpStatvfsPacket struct {
	ID   uint32
	Path string
}

func (p sshFxpStatvfsPacket) id() uint32 { return p.ID }

func (p sshFxpStatvfsPacket) MarshalBinary() ([]byte, error) {
	l := 1 + 4 + // type(byte) + uint32
		len(p.Path) +
		len("statvfs@openssh.com")

	b := make([]byte, 0, l)
	b = append(b, ssh_FXP_EXTENDED)
	b = marshalUint32(b, p.ID)
	b = marshalString(b, "statvfs@openssh.com")
	b = marshalString(b, p.Path)
	return b, nil
}

// A StatVFS contains statistics about a filesystem.
type StatVFS struct {
	ID      uint32
	Bsize   uint64 /* file system block size */
	Frsize  uint64 /* fundamental fs block size */
	Blocks  uint64 /* number of blocks (unit f_frsize) */
	Bfree   uint64 /* free blocks in file system */
	Bavail  uint64 /* free blocks for non-root */
	Files   uint64 /* total file inodes */
	Ffree   uint64 /* free file inodes */
	Favail  uint64 /* free file inodes for to non-root */
	Fsid    uint64 /* file system id */
	Flag    uint64 /* bit mask of f_flag values */
	Namemax uint64 /* maximum filename length */
}

// TotalSpace calculates the amount of total space in a filesystem.
func (p *StatVFS) TotalSpace() uint64 {
	return p.Frsize * p.Blocks
}

// FreeSpace calculates the amount of free space in a filesystem.
func (p *StatVFS) FreeSpace() uint64 {
	return p.Frsize * p.Bfree
}

// Convert to ssh_FXP_EXTENDED_REPLY packet binary format
func (p *StatVFS) MarshalBinary() ([]byte, error) {
	var buf bytes.Buffer
	buf.Write([]byte{ssh_FXP_EXTENDED_REPLY})
	err := binary.Write(&buf, binary.BigEndian, p)
	return buf.Bytes(), err
}

type sshFxpExtendedPacket struct {
	ID              uint32
	ExtendedRequest string
	SpecificPacket  interface {
		serverRespondablePacket
		readonly() bool
	}
}

func (p sshFxpExtendedPacket) id() uint32     { return p.ID }
func (p sshFxpExtendedPacket) readonly() bool { return p.SpecificPacket.readonly() }

func (p sshFxpExtendedPacket) respond(svr *Server) error {
	return p.SpecificPacket.respond(svr)
}

func (p *sshFxpExtendedPacket) UnmarshalBinary(b []byte) error {
	var err error
	bOrig := b
	if p.ID, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	} else if p.ExtendedRequest, b, err = unmarshalStringSafe(b); err != nil {
		return err
	}

	// specific unmarshalling
	switch p.ExtendedRequest {
	case "statvfs@openssh.com":
		p.SpecificPacket = &sshFxpExtendedPacketStatVFS{}
	default:
		return errUnknownExtendedPacket
	}

	return p.SpecificPacket.UnmarshalBinary(bOrig)
}

type sshFxpExtendedPacketStatVFS struct {
	ID              uint32
	ExtendedRequest string
	Path            string
}

func (p sshFxpExtendedPacketStatVFS) id() uint32     { return p.ID }
func (p sshFxpExtendedPacketStatVFS) readonly() bool { return true }
func (p *sshFxpExtendedPacketStatVFS) UnmarshalBinary(b []byte) error {
	var err error
	if p.ID, b, err = unmarshalUint32Safe(b); err != nil {
		return err
	} else if p.ExtendedRequest, b, err = unmarshalStringSafe(b); err != nil {
		return err
	} else if p.Path, b, err = unmarshalStringSafe(b); err != nil {
		return err
	}
	return nil
}
