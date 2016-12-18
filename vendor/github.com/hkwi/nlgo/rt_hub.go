// +build linux

package nlgo

import (
	"log"
	"sync"
	"syscall"
)

type NetlinkListener interface {
	NetlinkListen(syscall.NetlinkMessage)
}

// RtHub is a high layer thread-safe API, which is not present in libnl.
// RtHub is useful for listening to kernel event notification.
type RtHub struct {
	sock      *NlSock
	lock      *sync.Mutex
	unilock   *sync.Mutex
	uniseq    uint32
	unicast   NetlinkListener
	multicast map[uint32][]NetlinkListener
}

func NewRtHub() (*RtHub, error) {
	self := &RtHub{
		sock:      NlSocketAlloc(),
		lock:      &sync.Mutex{},
		unilock:   &sync.Mutex{},
		multicast: make(map[uint32][]NetlinkListener),
	}
	if err := NlConnect(self.sock, syscall.NETLINK_ROUTE); err != nil {
		NlSocketFree(self.sock)
		return nil, err
	}
	go func() {
		for {
			buf := make([]byte, syscall.Getpagesize())
			if n, _, err := syscall.Recvfrom(self.sock.Fd, buf, syscall.MSG_TRUNC); err != nil {
				if e, ok := err.(syscall.Errno); ok && e.Temporary() {
					continue
				}
				break
			} else if msgs, err := syscall.ParseNetlinkMessage(buf[:n]); err != nil {
				break
			} else {
				for _, msg := range msgs {
					multi := func() []NetlinkListener {
						self.lock.Lock()
						defer self.lock.Unlock()

						var ret []NetlinkListener
						for _, s := range self.multicast {
							ret = append(ret, s...)
						}
						return ret
					}()
					if msg.Header.Seq == self.uniseq {
						if self.unicast != nil {
							self.unicast.NetlinkListen(msg)
						}
						switch msg.Header.Type {
						case syscall.NLMSG_DONE, syscall.NLMSG_ERROR:
							self.unilock.Unlock()
						}
					}
					if msg.Header.Seq == 0 {
						for _, proc := range multi {
							proc.NetlinkListen(msg)
						}
					}
				}
			}
		}
		log.Print("rt hub loop exit")
	}()
	return self, nil
}

func (self RtHub) Close() {
	NlSocketFree(self.sock)
}

// Async() submits request with callback. Note that this locks sending request.
// Calling Async() in GenlListen() may create dead lock.
// netlink message header will be reparsed.
func (self *RtHub) Async(msg syscall.NetlinkMessage, listener NetlinkListener) error {
	self.unilock.Lock()
	self.unicast = listener
	self.uniseq = self.sock.SeqNext

	hdr := msg.Header
	if err := NlSendSimple(self.sock, hdr.Type, hdr.Flags, msg.Data); err != nil {
		self.unilock.Unlock()
		return err
	}
	return nil
}

type hubCapture struct {
	Msgs []syscall.NetlinkMessage
}

func (self *hubCapture) NetlinkListen(msg syscall.NetlinkMessage) {
	self.Msgs = append(self.Msgs, msg)
}

// Sync() is synchronous version of Async().
// Calling Sync() in GenlListen() may create dead lock.
func (self *RtHub) Sync(msg syscall.NetlinkMessage) ([]syscall.NetlinkMessage, error) {
	cap := &hubCapture{}
	if err := self.Async(msg, cap); err != nil {
		return nil, err
	} else {
		self.unilock.Lock() // waits for unlock, which means response arrival.
		defer self.unilock.Unlock()

		return cap.Msgs, nil
	}
}

// Add adds a listener to the hub.
// listener will recieve all of the rtnetlink events, regardless of their group registration.
// If you want to split it, then use separate RtHub.
func (self RtHub) Add(group uint32, listener NetlinkListener) error {
	self.lock.Lock()
	defer self.lock.Unlock()

	if len(self.multicast[group]) == 0 {
		if err := NlSocketAddMembership(self.sock, int(group)); err != nil {
			return err
		}
	}
	self.multicast[group] = append(self.multicast[group], listener)
	return nil
}

func (self RtHub) Remove(group uint32, listener NetlinkListener) error {
	self.lock.Lock()
	defer self.lock.Unlock()

	var active []NetlinkListener
	for _, li := range self.multicast[group] {
		if li != listener {
			active = append(active, li)
		}
	}
	self.multicast[group] = active

	if len(active) == 0 {
		if err := NlSocketDropMembership(self.sock, int(group)); err != nil {
			return err
		}
	}
	return nil
}
