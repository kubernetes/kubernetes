package main

import (
	"github.com/hkwi/nlgo"
	"log"
	"syscall"
	"unsafe"
)

func main() {
	sk := nlgo.NlSocketAlloc()
	if err := nlgo.GenlConnect(sk); err != nil {
		panic(err)
	} else if err := nlgo.GenlSendSimple(sk, nlgo.GENL_ID_CTRL, nlgo.CTRL_CMD_GETFAMILY, nlgo.CTRL_VERSION, syscall.NLM_F_DUMP); err != nil {
		panic(err)
	}
	nl80211 := nlgo.NlSocketAlloc()
	if err := nlgo.GenlConnect(nl80211); err != nil {
		panic(err)
	}

	data := make([]byte, syscall.Getpagesize())
	if n, _, _, _, err := syscall.Recvmsg(sk.Fd, data, nil, 0); err != nil {
		panic(err)
	} else if msgs, err := syscall.ParseNetlinkMessage(data[:n]); err != nil {
		log.Print("X", err)
	} else {
		for _, msg := range msgs {
			genl := *(*nlgo.GenlMsghdr)(unsafe.Pointer(&msg.Data[0]))
			if msg.Header.Type == nlgo.GENL_ID_CTRL && genl.Cmd == nlgo.CTRL_CMD_NEWFAMILY {
				if attr, err := nlgo.CtrlPolicy.Parse(msg.Data[nlgo.GENL_HDRLEN:]); err != nil {
					log.Print(err)
				} else if amap, ok := attr.(nlgo.AttrMap); !ok {
					log.Print(attr)
				} else if value := amap.Get(nlgo.CTRL_ATTR_FAMILY_NAME).(nlgo.NulString); string(value) == "nl80211" {
					log.Printf("%v", attr)
					for _, g := range amap.Get(nlgo.CTRL_ATTR_MCAST_GROUPS).(nlgo.AttrSlice).Slice() {
						group := g.Value.(nlgo.AttrMap)
						pid := group.Get(nlgo.CTRL_ATTR_MCAST_GRP_ID).(nlgo.U32)
						if err := nlgo.NlSocketAddMembership(nl80211, int(pid)); err != nil {
							log.Print(err)
						}
					}
				}
			} else {
				log.Print("UNKNOWN")
			}
		}
	}
	nlgo.NlSocketFree(sk)

	for {
		if n, _, _, _, err := syscall.Recvmsg(nl80211.Fd, data, nil, 0); err != nil {
			panic(err)
		} else if msgs, err := syscall.ParseNetlinkMessage(data[:n]); err != nil {
			log.Print("Y", err)
		} else {
			for _, msg := range msgs {
				genl := (*nlgo.GenlMsghdr)(unsafe.Pointer(&msg.Data[0]))
				if attr, err := nlgo.Nl80211Policy.Parse(msg.Data[nlgo.GENL_HDRLEN:]); err != nil {
					log.Print("Z", err)
				} else {
					log.Printf("NL80211_CMD_%s attrs=%s", nlgo.NL80211_CMD_itoa[genl.Cmd], attr)
				}
			}
		}
	}
}
