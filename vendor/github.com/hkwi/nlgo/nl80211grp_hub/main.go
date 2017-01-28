package main

import (
	"github.com/hkwi/nlgo"
	"log"
	"syscall"
)

type capture struct{}

func (self capture) GenlListen(msg nlgo.GenlMessage) {
	switch msg.Header.Type {
	case syscall.NLMSG_DONE:
		log.Print("Init DONE")
	case syscall.NLMSG_ERROR:
		log.Print(nlgo.NlMsgerr(msg.NetlinkMessage))
	default:
		if attrs, err := nlgo.Nl80211Policy.Parse(msg.Body()); err != nil {
			panic(err)
		} else {
			log.Printf("NL80211_CMD_%s attrs=%s", nlgo.NL80211_CMD_itoa[msg.Genl().Cmd], attrs)
		}
	}
}

func main() {
	cap := capture{}

	ghub, e1 := nlgo.NewGenlHub()
	if e1 != nil {
		panic(e1)
	}
	nl80211 := ghub.Family("nl80211")

	msgs, e2 := ghub.Sync(nlgo.GenlFamilyCtrl.DumpRequest(nlgo.CTRL_CMD_GETFAMILY))
	if e2 != nil {
		panic(e2)
	}
	for _, msg := range msgs {
		switch msg.Header.Type {
		case syscall.NLMSG_DONE:
			// do nothing
		case syscall.NLMSG_ERROR:
			log.Print(nlgo.NlMsgerr(msg.NetlinkMessage))
		case nlgo.GENL_ID_CTRL:
			if family, groups, e3 := nlgo.GenlCtrl(nlgo.GenlFamilyCtrl).Parse(msg); e3 != nil {
				panic(e3)
			} else if family.Name != "nl80211" {
				continue
			} else if msg.Genl().Cmd == nlgo.CTRL_CMD_NEWFAMILY {
				for _, group := range groups {
					if e4 := ghub.Add("nl80211", group.Name, cap); e4 != nil {
						panic(e4)
					}
				}
			}
		}
	}

	if err := ghub.Async(nl80211.DumpRequest(nlgo.NL80211_CMD_GET_WIPHY), cap); err != nil {
		panic(err)
	}
	wait := make(chan bool)
	<-wait
}
