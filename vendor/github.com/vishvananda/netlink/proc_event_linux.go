package netlink

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"syscall"

	"github.com/vishvananda/netlink/nl"
	"github.com/vishvananda/netns"
	"golang.org/x/sys/unix"
)

const CN_IDX_PROC = 0x1

const (
	PROC_EVENT_NONE     = 0x00000000
	PROC_EVENT_FORK     = 0x00000001
	PROC_EVENT_EXEC     = 0x00000002
	PROC_EVENT_UID      = 0x00000004
	PROC_EVENT_GID      = 0x00000040
	PROC_EVENT_SID      = 0x00000080
	PROC_EVENT_PTRACE   = 0x00000100
	PROC_EVENT_COMM     = 0x00000200
	PROC_EVENT_COREDUMP = 0x40000000
	PROC_EVENT_EXIT     = 0x80000000
)

const (
	CN_VAL_PROC          = 0x1
	PROC_CN_MCAST_LISTEN = 0x1
)

type ProcEventMsg interface {
	Pid() uint32
	Tgid() uint32
}

type ProcEventHeader struct {
	What      uint32
	CPU       uint32
	Timestamp uint64
}

type ProcEvent struct {
	ProcEventHeader
	Msg ProcEventMsg
}

func (pe *ProcEvent) setHeader(h ProcEventHeader) {
	pe.What = h.What
	pe.CPU = h.CPU
	pe.Timestamp = h.Timestamp
}

type ExitProcEvent struct {
	ProcessPid  uint32
	ProcessTgid uint32
	ExitCode    uint32
	ExitSignal  uint32
	ParentPid   uint32
	ParentTgid  uint32
}

func (e *ExitProcEvent) Pid() uint32 {
	return e.ProcessPid
}

func (e *ExitProcEvent) Tgid() uint32 {
	return e.ProcessTgid
}

type ExecProcEvent struct {
	ProcessPid  uint32
	ProcessTgid uint32
}

func (e *ExecProcEvent) Pid() uint32 {
	return e.ProcessPid
}

func (e *ExecProcEvent) Tgid() uint32 {
	return e.ProcessTgid
}

type ForkProcEvent struct {
	ParentPid  uint32
	ParentTgid uint32
	ChildPid   uint32
	ChildTgid  uint32
}

func (e *ForkProcEvent) Pid() uint32 {
	return e.ParentPid
}

func (e *ForkProcEvent) Tgid() uint32 {
	return e.ParentTgid
}

type CommProcEvent struct {
	ProcessPid  uint32
	ProcessTgid uint32
	Comm        [16]byte
}

func (e *CommProcEvent) Pid() uint32 {
	return e.ProcessPid
}

func (e *CommProcEvent) Tgid() uint32 {
	return e.ProcessTgid
}

func ProcEventMonitor(ch chan<- ProcEvent, done <-chan struct{}, errorChan chan<- error) error {
	h, err := NewHandle()
	if err != nil {
		return err
	}
	defer h.Delete()

	s, err := nl.SubscribeAt(netns.None(), netns.None(), unix.NETLINK_CONNECTOR, CN_IDX_PROC)
	if err != nil {
		return err
	}

	var nlmsg nl.NetlinkRequest

	nlmsg.Pid = uint32(os.Getpid())
	nlmsg.Type = unix.NLMSG_DONE
	nlmsg.Len = uint32(unix.SizeofNlMsghdr)

	cm := nl.NewCnMsg(CN_IDX_PROC, CN_VAL_PROC, PROC_CN_MCAST_LISTEN)
	nlmsg.AddData(cm)

	s.Send(&nlmsg)

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
				e := parseNetlinkMessage(m)
				if e != nil {
					ch <- *e
				}
			}

		}
	}()

	return nil
}

func parseNetlinkMessage(m syscall.NetlinkMessage) *ProcEvent {
	if m.Header.Type == unix.NLMSG_DONE {
		buf := bytes.NewBuffer(m.Data)
		msg := &nl.CnMsg{}
		hdr := &ProcEventHeader{}
		binary.Read(buf, nl.NativeEndian(), msg)
		binary.Read(buf, nl.NativeEndian(), hdr)

		pe := &ProcEvent{}
		pe.setHeader(*hdr)
		switch hdr.What {
		case PROC_EVENT_EXIT:
			event := &ExitProcEvent{}
			binary.Read(buf, nl.NativeEndian(), event)
			pe.Msg = event
			return pe
		case PROC_EVENT_FORK:
			event := &ForkProcEvent{}
			binary.Read(buf, nl.NativeEndian(), event)
			pe.Msg = event
			return pe
		case PROC_EVENT_EXEC:
			event := &ExecProcEvent{}
			binary.Read(buf, nl.NativeEndian(), event)
			pe.Msg = event
			return pe
		case PROC_EVENT_COMM:
			event := &CommProcEvent{}
			binary.Read(buf, nl.NativeEndian(), event)
			pe.Msg = event
			return pe
		}
		return nil
	}

	return nil
}
