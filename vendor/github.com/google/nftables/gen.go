package nftables

import (
	"encoding/binary"
	"fmt"
	"github.com/mdlayher/netlink"
	"golang.org/x/sys/unix"
)

type GenMsg struct {
	ID       uint32
	ProcPID  uint32
	ProcComm string // [16]byte - max 16bytes - kernel TASK_COMM_LEN
}

var genHeaderType = netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_NEWGEN)

func genFromMsg(msg netlink.Message) (*GenMsg, error) {
	if got, want := msg.Header.Type, genHeaderType; got != want {
		return nil, fmt.Errorf("unexpected header type: got %v, want %v", got, want)
	}
	ad, err := netlink.NewAttributeDecoder(msg.Data[4:])
	if err != nil {
		return nil, err
	}
	ad.ByteOrder = binary.BigEndian

	msgOut := &GenMsg{}
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_GEN_ID:
			msgOut.ID = ad.Uint32()
		case unix.NFTA_GEN_PROC_PID:
			msgOut.ProcPID = ad.Uint32()
		case unix.NFTA_GEN_PROC_NAME:
			msgOut.ProcComm = ad.String()
		default:
			return nil, fmt.Errorf("Unknown attribute: %d %v\n", ad.Type(), ad.Bytes())
		}
	}
	if err := ad.Err(); err != nil {
		return nil, err
	}
	return msgOut, nil
}
