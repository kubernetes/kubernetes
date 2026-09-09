package netlink

import (
	"fmt"

	"github.com/vishvananda/netlink/nl"
	"github.com/vishvananda/netns"
	"golang.org/x/sys/unix"
)

type XfrmMsg interface {
	Type() nl.XfrmMsgType
}

type XfrmMsgExpire struct {
	XfrmState *XfrmState
	Hard      bool
}

func (ue *XfrmMsgExpire) Type() nl.XfrmMsgType {
	return nl.XFRM_MSG_EXPIRE
}

func parseXfrmMsgExpire(b []byte) *XfrmMsgExpire {
	var e XfrmMsgExpire

	msg := nl.DeserializeXfrmUserExpire(b)
	e.XfrmState = xfrmStateFromXfrmUsersaInfo(&msg.XfrmUsersaInfo)
	e.Hard = msg.Hard == 1

	return &e
}

func XfrmMonitor(ch chan<- XfrmMsg, done <-chan struct{}, errorChan chan<- error,
	types ...nl.XfrmMsgType) error {

	groups, err := xfrmMcastGroups(types)
	if err != nil {
		return nil
	}
	s, err := nl.SubscribeAt(netns.None(), netns.None(), unix.NETLINK_XFRM, groups...)
	if err != nil {
		return err
	}

	if done != nil {
		go func() {
			<-done
			s.Close()
		}()

	}

	go func() {
		defer close(ch)
		for {
			msgs, from, err := s.Receive()
			if err != nil {
				errorChan <- err
				return
			}
			if from.Pid != nl.PidKernel {
				errorChan <- fmt.Errorf("Wrong sender portid %d, expected %d", from.Pid, nl.PidKernel)
				return
			}
			for _, m := range msgs {
				switch m.Header.Type {
				case nl.XFRM_MSG_EXPIRE:
					ch <- parseXfrmMsgExpire(m.Data)
				default:
					errorChan <- fmt.Errorf("unsupported msg type: %x", m.Header.Type)
				}
			}
		}
	}()

	return nil
}

func xfrmMcastGroups(types []nl.XfrmMsgType) ([]uint, error) {
	groups := make([]uint, 0)

	if len(types) == 0 {
		return nil, fmt.Errorf("no xfrm msg type specified")
	}

	for _, t := range types {
		var group uint

		switch t {
		case nl.XFRM_MSG_EXPIRE:
			group = nl.XFRMNLGRP_EXPIRE
		default:
			return nil, fmt.Errorf("unsupported group: %x", t)
		}

		groups = append(groups, group)
	}

	return groups, nil
}
