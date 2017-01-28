// rtlink provides RTM_*LINK util
package rtlink

import (
	"fmt"
	"github.com/hkwi/nlgo"
	"syscall"
)

func GetByName(hub *nlgo.RtHub, name string) (syscall.IfInfomsg, error) {
	var ret syscall.IfInfomsg

	req := &syscall.NetlinkMessage{
		Header: syscall.NlMsghdr{
			Type:  syscall.RTM_GETLINK,
			Flags: syscall.NLM_F_REQUEST,
		},
	}
	(*nlgo.IfInfoMessage)(req).Set(
		syscall.IfInfomsg{},
		nlgo.AttrSlice{
			nlgo.Attr{
				Header: syscall.NlAttr{
					Type: syscall.IFLA_IFNAME,
				},
				Value: nlgo.NulString(name),
			},
		})

	if msgs, err := hub.Sync(*req); err != nil {
		return ret, err
	} else {
		for _, msg := range msgs {
			switch msg.Header.Type {
			case syscall.RTM_NEWLINK:
				info := (nlgo.IfInfoMessage)(msg)
				if attrs, err := info.Attrs(); err != nil {
					continue
				} else if string(attrs.(nlgo.AttrMap).Get(syscall.IFLA_IFNAME).(nlgo.NulString)) == name {
					return info.IfInfo(), nil
				}
			}
		}
	}
	return ret, fmt.Errorf("response empty")
}

func GetNameByIndex(hub *nlgo.RtHub, index int) (string, error) {
	req := &syscall.NetlinkMessage{
		Header: syscall.NlMsghdr{
			Type:  syscall.RTM_GETLINK,
			Flags: syscall.NLM_F_REQUEST,
		},
	}
	(*nlgo.IfInfoMessage)(req).Set(
		syscall.IfInfomsg{
			Index: int32(index),
		},
		nil)

	if msgs, err := hub.Sync(*req); err != nil {
		return "", err
	} else {
		for _, msg := range msgs {
			switch msg.Header.Type {
			case syscall.RTM_NEWLINK:
				info := (nlgo.IfInfoMessage)(msg)
				if info.IfInfo().Index == int32(index) {
					if attrs, err := info.Attrs(); err != nil {
						return "", err
					} else {
						return string(attrs.(nlgo.AttrMap).Get(syscall.IFLA_IFNAME).(nlgo.NulString)), nil
					}
				}
			}
		}
		return "", fmt.Errorf("response empty")
	}
}
