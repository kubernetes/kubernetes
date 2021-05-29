// +build linux

package netlink

import (
	"encoding/binary"
	"errors"

	"github.com/vishvananda/netlink/nl"
	"golang.org/x/sys/unix"
)

const (
	FOU_GENL_NAME = "fou"
)

const (
	FOU_CMD_UNSPEC uint8 = iota
	FOU_CMD_ADD
	FOU_CMD_DEL
	FOU_CMD_GET
	FOU_CMD_MAX = FOU_CMD_GET
)

const (
	FOU_ATTR_UNSPEC = iota
	FOU_ATTR_PORT
	FOU_ATTR_AF
	FOU_ATTR_IPPROTO
	FOU_ATTR_TYPE
	FOU_ATTR_REMCSUM_NOPARTIAL
	FOU_ATTR_MAX = FOU_ATTR_REMCSUM_NOPARTIAL
)

const (
	FOU_ENCAP_UNSPEC = iota
	FOU_ENCAP_DIRECT
	FOU_ENCAP_GUE
	FOU_ENCAP_MAX = FOU_ENCAP_GUE
)

var fouFamilyId int

func FouFamilyId() (int, error) {
	if fouFamilyId != 0 {
		return fouFamilyId, nil
	}

	fam, err := GenlFamilyGet(FOU_GENL_NAME)
	if err != nil {
		return -1, err
	}

	fouFamilyId = int(fam.ID)
	return fouFamilyId, nil
}

func FouAdd(f Fou) error {
	return pkgHandle.FouAdd(f)
}

func (h *Handle) FouAdd(f Fou) error {
	fam_id, err := FouFamilyId()
	if err != nil {
		return err
	}

	// setting ip protocol conflicts with encapsulation type GUE
	if f.EncapType == FOU_ENCAP_GUE && f.Protocol != 0 {
		return errors.New("GUE encapsulation doesn't specify an IP protocol")
	}

	req := h.newNetlinkRequest(fam_id, unix.NLM_F_ACK)

	// int to byte for port
	bp := make([]byte, 2)
	binary.BigEndian.PutUint16(bp[0:2], uint16(f.Port))

	attrs := []*nl.RtAttr{
		nl.NewRtAttr(FOU_ATTR_PORT, bp),
		nl.NewRtAttr(FOU_ATTR_TYPE, []byte{uint8(f.EncapType)}),
		nl.NewRtAttr(FOU_ATTR_AF, []byte{uint8(f.Family)}),
		nl.NewRtAttr(FOU_ATTR_IPPROTO, []byte{uint8(f.Protocol)}),
	}
	raw := []byte{FOU_CMD_ADD, 1, 0, 0}
	for _, a := range attrs {
		raw = append(raw, a.Serialize()...)
	}

	req.AddRawData(raw)

	_, err = req.Execute(unix.NETLINK_GENERIC, 0)
	return err
}

func FouDel(f Fou) error {
	return pkgHandle.FouDel(f)
}

func (h *Handle) FouDel(f Fou) error {
	fam_id, err := FouFamilyId()
	if err != nil {
		return err
	}

	req := h.newNetlinkRequest(fam_id, unix.NLM_F_ACK)

	// int to byte for port
	bp := make([]byte, 2)
	binary.BigEndian.PutUint16(bp[0:2], uint16(f.Port))

	attrs := []*nl.RtAttr{
		nl.NewRtAttr(FOU_ATTR_PORT, bp),
		nl.NewRtAttr(FOU_ATTR_AF, []byte{uint8(f.Family)}),
	}
	raw := []byte{FOU_CMD_DEL, 1, 0, 0}
	for _, a := range attrs {
		raw = append(raw, a.Serialize()...)
	}

	req.AddRawData(raw)

	_, err = req.Execute(unix.NETLINK_GENERIC, 0)
	if err != nil {
		return err
	}

	return nil
}

func FouList(fam int) ([]Fou, error) {
	return pkgHandle.FouList(fam)
}

func (h *Handle) FouList(fam int) ([]Fou, error) {
	fam_id, err := FouFamilyId()
	if err != nil {
		return nil, err
	}

	req := h.newNetlinkRequest(fam_id, unix.NLM_F_DUMP)

	attrs := []*nl.RtAttr{
		nl.NewRtAttr(FOU_ATTR_AF, []byte{uint8(fam)}),
	}
	raw := []byte{FOU_CMD_GET, 1, 0, 0}
	for _, a := range attrs {
		raw = append(raw, a.Serialize()...)
	}

	req.AddRawData(raw)

	msgs, err := req.Execute(unix.NETLINK_GENERIC, 0)
	if err != nil {
		return nil, err
	}

	fous := make([]Fou, 0, len(msgs))
	for _, m := range msgs {
		f, err := deserializeFouMsg(m)
		if err != nil {
			return fous, err
		}

		fous = append(fous, f)
	}

	return fous, nil
}

func deserializeFouMsg(msg []byte) (Fou, error) {
	// we'll skip to byte 4 to first attribute
	msg = msg[3:]
	var shift int
	fou := Fou{}

	for {
		// attribute header is at least 16 bits
		if len(msg) < 4 {
			return fou, ErrAttrHeaderTruncated
		}

		lgt := int(binary.BigEndian.Uint16(msg[0:2]))
		if len(msg) < lgt+4 {
			return fou, ErrAttrBodyTruncated
		}
		attr := binary.BigEndian.Uint16(msg[2:4])

		shift = lgt + 3
		switch attr {
		case FOU_ATTR_AF:
			fou.Family = int(msg[5])
		case FOU_ATTR_PORT:
			fou.Port = int(binary.BigEndian.Uint16(msg[5:7]))
			// port is 2 bytes
			shift = lgt + 2
		case FOU_ATTR_IPPROTO:
			fou.Protocol = int(msg[5])
		case FOU_ATTR_TYPE:
			fou.EncapType = int(msg[5])
		}

		msg = msg[shift:]

		if len(msg) < 4 {
			break
		}
	}

	return fou, nil
}
