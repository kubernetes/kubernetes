// +build linux

package nlgo

import (
	"fmt"
	"log"
	"sync"
	"syscall"
	"unsafe"
)

type GenlMessage struct {
	syscall.NetlinkMessage
	Family GenlFamily
}

func (self GenlMessage) Genl() GenlMsghdr {
	return *(*GenlMsghdr)(unsafe.Pointer(&self.Data[0]))
}

func (self GenlMessage) FamilyHeader() []byte {
	return self.Data[GENL_HDRLEN : GENL_HDRLEN+NLMSG_ALIGN(int(self.Family.Hdrsize))]
}

func (self GenlMessage) Body() []byte {
	return self.Data[GENL_HDRLEN+NLMSG_ALIGN(int(self.Family.Hdrsize)):]
}

type GenlListener interface {
	GenlListen(GenlMessage)
}

type groupKey struct {
	Family string
	Name   string
}

// GenlHub is generic netlink version of RtHub.
type GenlHub struct {
	sock       *NlSock
	lock       *sync.Mutex
	familyIds  map[uint16]GenlFamily
	groupIds   map[uint32]GenlGroup
	membership []uint32
	unilock    *sync.Mutex
	uniseq     uint32
	unicast    GenlListener
	multicast  map[groupKey][]GenlListener
}

// sync group registeration
func (self *GenlHub) sync() error {
	var active []uint32
	for gkey, _ := range self.multicast {
		for _, ginfo := range self.groupIds {
			if ginfo.Family == gkey.Family && ginfo.Name == gkey.Name {
				active = append(active, ginfo.Id)
			}
		}
	}
	for _, old := range self.membership {
		// should not happen because CTRL_CMD_DELMCAST_GRP event have already cleaned up.
		drop := true
		for _, new := range active {
			if new == old {
				drop = false
			}
		}
		if drop {
			if err := NlSocketDropMembership(self.sock, int(old)); err != nil {
				return err
			}
		}
	}
	for _, new := range active {
		join := true
		for _, old := range self.membership {
			if new == old {
				join = false
			}
		}
		if join {
			if err := NlSocketAddMembership(self.sock, int(new)); err != nil {
				return err
			}
		}
	}
	self.membership = active
	return nil
}

func NewGenlHub() (*GenlHub, error) {
	self := &GenlHub{
		sock: NlSocketAlloc(),
		lock: &sync.Mutex{},
		familyIds: map[uint16]GenlFamily{
			GENL_ID_CTRL: GenlFamilyCtrl,
		},
		groupIds: map[uint32]GenlGroup{
			GENL_ID_CTRL: GenlGroup{
				Id:     GENL_ID_CTRL,
				Family: "nlctrl",
				Name:   "notify",
			},
		},
		unilock:   &sync.Mutex{},
		multicast: make(map[groupKey][]GenlListener),
	}
	if err := NlConnect(self.sock, syscall.NETLINK_GENERIC); err != nil {
		self.Close()
		return nil, err
	}
	if err := self.Add("nlctrl", "notify", self); err != nil {
		self.Close()
		return nil, err
	}
	if err := self.Async(GenlFamilyCtrl.DumpRequest(CTRL_CMD_GETFAMILY), self); err != nil {
		self.Close()
		return nil, err
	}
	go func() {
		for {
			buf := make([]byte, syscall.Getpagesize())
			if bufN, _, err := syscall.Recvfrom(self.sock.Fd, buf, syscall.MSG_TRUNC); err != nil {
				if e, ok := err.(syscall.Errno); ok && e.Temporary() {
					continue
				}
				break
			} else if msgs, err := syscall.ParseNetlinkMessage(buf[:bufN]); err != nil {
				break
			} else {
				for _, msg := range msgs {
					self.lock.Lock()
					family := self.familyIds[msg.Header.Type]
					var multi []GenlListener
					for gkey, s := range self.multicast {
						if family.Name == gkey.Family {
							for _, n := range s {
								if uniq := func() bool {
									for _, m := range multi {
										if m == n {
											return false
										}
									}
									return true
								}(); uniq {
									multi = append(multi, n)
								}
							}
						}
					}
					self.lock.Unlock()

					gmsg := GenlMessage{
						NetlinkMessage: msg,
						Family:         family,
					}
					if msg.Header.Seq == self.uniseq {
						if self.unicast != nil {
							self.unicast.GenlListen(gmsg)
						}
						switch msg.Header.Type {
						case syscall.NLMSG_DONE, syscall.NLMSG_ERROR:
							self.unilock.Unlock()
						}
					}
					if msg.Header.Seq == 0 {
						for _, proc := range multi {
							proc.GenlListen(gmsg)
						}
					}
				}
			}
		}
		log.Print("genl hub loop exit")
	}()
	return self, nil
}

func (self GenlHub) Close() {
	NlSocketFree(self.sock)
}

// Async() submits request with callback. Note that this locks sending request.
// Calling Async() in GenlListen() may create dead lock.
func (self *GenlHub) Async(msg GenlMessage, listener GenlListener) error {
	self.unilock.Lock()

	self.lock.Lock()
	defer self.lock.Unlock()

	self.unicast = listener
	self.uniseq = self.sock.SeqNext
	hdr := msg.Header

	if err := NlSendSimple(self.sock, hdr.Type, hdr.Flags, msg.Data); err != nil {
		self.unilock.Unlock()
		return err
	}
	return nil
}

type genlHubCapture struct {
	Msgs []GenlMessage
}

func (self *genlHubCapture) GenlListen(msg GenlMessage) {
	self.Msgs = append(self.Msgs, msg)
}

// Sync() is synchronous version of Async().
// Calling Sync() in GenlListen() may create dead lock.
func (self *GenlHub) Sync(msg GenlMessage) ([]GenlMessage, error) {
	cap := &genlHubCapture{}
	if err := self.Async(msg, cap); err != nil {
		return nil, err
	} else {
		self.unilock.Lock() // wait for unicast processing
		defer self.unilock.Unlock()
		return cap.Msgs, nil
	}
}

func (self GenlHub) GenlListen(msg GenlMessage) {
	if msg.Header.Type != GENL_ID_CTRL {
		return
	}
	if family, groups, err := GenlCtrl(GenlFamilyCtrl).Parse(msg); err != nil {
		log.Printf("genl ctrl msg parse err %v", err)
	} else {
		self.lock.Lock()
		defer self.lock.Unlock()

		switch msg.Genl().Cmd {
		case CTRL_CMD_NEWFAMILY:
			self.familyIds[family.Id] = family
			fallthrough
		case CTRL_CMD_NEWMCAST_GRP:
			for _, grp := range groups {
				if _, exists := self.groupIds[grp.Id]; !exists {
					self.groupIds[grp.Id] = grp
				}
			}
		case CTRL_CMD_DELFAMILY:
			delete(self.familyIds, family.Id)
			fallthrough
		case CTRL_CMD_DELMCAST_GRP:
			for _, grp := range groups {
				if _, exists := self.groupIds[grp.Id]; exists {
					delete(self.groupIds, grp.Id)
				}
			}
		}
		self.sync()
	}
}

// Add adds a GenlListener to GenlHub.
// listeners will recieve all of the same family events, regardless of their group registration.
// If you want to limited group multicast, create separate GenlHub for each.
func (self GenlHub) Add(family, group string, listener GenlListener) error {
	self.lock.Lock()
	defer self.lock.Unlock()

	key := groupKey{Family: family, Name: group}
	self.multicast[key] = append(self.multicast[key], listener)
	return self.sync()
}

func (self GenlHub) Remove(family, group string, listener GenlListener) error {
	self.lock.Lock()
	defer self.lock.Unlock()

	key := groupKey{Family: family, Name: group}
	var active []GenlListener
	for _, li := range self.multicast[key] {
		if li != listener {
			active = append(active, li)
		}
	}
	self.multicast[key] = active
	return self.sync()
}

type familyListener struct {
	name string
	done GenlFamily
}

func (self *familyListener) GenlListen(msg GenlMessage) {
	switch msg.Header.Type {
	case GENL_ID_CTRL:
		if family, _, err := GenlCtrl(GenlFamilyCtrl).Parse(msg); err != nil {
			log.Printf("genl ctrl %v", err)
		} else if family.Name == self.name {
			self.done = family
		}
	}
}

// Family waits for genl family loading. Make sure loading that module by modprobe or by other means.
func (self *GenlHub) Family(name string) GenlFamily {
	cap := &familyListener{name: name}
	self.Add("nlctrl", "notify", cap)
	defer self.Remove("nlctrl", "notify", cap)

	for _, f := range self.familyIds {
		if f.Name == name {
			return f
		}
	}
	if err := self.Async(GenlFamilyCtrl.DumpRequest(CTRL_CMD_GETFAMILY), cap); err != nil {
		log.Print(err)
	} else {
		self.unilock.Lock() // wait for unicast processing done.
		defer self.unilock.Unlock()
	}
	return cap.done
}

type GenlCtrl GenlFamily

func (self GenlCtrl) Parse(msg GenlMessage) (GenlFamily, []GenlGroup, error) {
	var ret GenlFamily
	if attrs, err := CtrlPolicy.Parse(msg.Body()); err != nil {
		return ret, nil, err
	} else if amap, ok := attrs.(AttrMap); !ok {
		return ret, nil, fmt.Errorf("genl ctrl policy parse error")
	} else {
		if t := amap.Get(CTRL_ATTR_FAMILY_ID); t != nil {
			ret.Id = uint16(t.(U16))
		}
		if t := amap.Get(CTRL_ATTR_FAMILY_NAME); t != nil {
			ret.Name = string(t.(NulString))
		}
		if t := amap.Get(CTRL_ATTR_VERSION); t != nil {
			ret.Version = uint8(uint32(t.(U32)))
		}
		if t := amap.Get(CTRL_ATTR_HDRSIZE); t != nil {
			ret.Hdrsize = uint32(t.(U32))
		}
		var groups []GenlGroup
		if grps := amap.Get(CTRL_ATTR_MCAST_GROUPS); grps != nil {
			for _, grp := range grps.(AttrSlice).Slice() {
				gattr := grp.Value.(AttrMap)
				key := uint32(gattr.Get(CTRL_ATTR_MCAST_GRP_ID).(U32))
				groups = append(groups, GenlGroup{
					Id:     key,
					Family: ret.Name,
					Name:   string(gattr.Get(CTRL_ATTR_MCAST_GRP_NAME).(NulString)),
				})
			}
		}
		return ret, groups, nil
	}
}
