package netlink

import (
	"fmt"
	"unsafe"

	"github.com/vishvananda/netlink/nl"
	"golang.org/x/sys/unix"
)

func writeStateAlgo(a *XfrmStateAlgo) []byte {
	algo := nl.XfrmAlgo{
		AlgKeyLen: uint32(len(a.Key) * 8),
		AlgKey:    a.Key,
	}
	end := len(a.Name)
	if end > 64 {
		end = 64
	}
	copy(algo.AlgName[:end], a.Name)
	return algo.Serialize()
}

func writeStateAlgoAuth(a *XfrmStateAlgo) []byte {
	algo := nl.XfrmAlgoAuth{
		AlgKeyLen:   uint32(len(a.Key) * 8),
		AlgTruncLen: uint32(a.TruncateLen),
		AlgKey:      a.Key,
	}
	end := len(a.Name)
	if end > 64 {
		end = 64
	}
	copy(algo.AlgName[:end], a.Name)
	return algo.Serialize()
}

func writeStateAlgoAead(a *XfrmStateAlgo) []byte {
	algo := nl.XfrmAlgoAEAD{
		AlgKeyLen: uint32(len(a.Key) * 8),
		AlgICVLen: uint32(a.ICVLen),
		AlgKey:    a.Key,
	}
	end := len(a.Name)
	if end > 64 {
		end = 64
	}
	copy(algo.AlgName[:end], a.Name)
	return algo.Serialize()
}

func writeMark(m *XfrmMark) []byte {
	mark := &nl.XfrmMark{
		Value: m.Value,
		Mask:  m.Mask,
	}
	if mark.Mask == 0 {
		mark.Mask = ^uint32(0)
	}
	return mark.Serialize()
}

func writeReplayEsn(replayWindow int) []byte {
	replayEsn := &nl.XfrmReplayStateEsn{
		OSeq:         0,
		Seq:          0,
		OSeqHi:       0,
		SeqHi:        0,
		ReplayWindow: uint32(replayWindow),
	}

	// taken from iproute2/ip/xfrm_state.c:
	replayEsn.BmpLen = uint32((replayWindow + (4 * 8) - 1) / (4 * 8))

	return replayEsn.Serialize()
}

// XfrmStateAdd will add an xfrm state to the system.
// Equivalent to: `ip xfrm state add $state`
func XfrmStateAdd(state *XfrmState) error {
	return pkgHandle.XfrmStateAdd(state)
}

// XfrmStateAdd will add an xfrm state to the system.
// Equivalent to: `ip xfrm state add $state`
func (h *Handle) XfrmStateAdd(state *XfrmState) error {
	return h.xfrmStateAddOrUpdate(state, nl.XFRM_MSG_NEWSA)
}

// XfrmStateAllocSpi will allocate an xfrm state in the system.
// Equivalent to: `ip xfrm state allocspi`
func XfrmStateAllocSpi(state *XfrmState) (*XfrmState, error) {
	return pkgHandle.xfrmStateAllocSpi(state)
}

// XfrmStateUpdate will update an xfrm state to the system.
// Equivalent to: `ip xfrm state update $state`
func XfrmStateUpdate(state *XfrmState) error {
	return pkgHandle.XfrmStateUpdate(state)
}

// XfrmStateUpdate will update an xfrm state to the system.
// Equivalent to: `ip xfrm state update $state`
func (h *Handle) XfrmStateUpdate(state *XfrmState) error {
	return h.xfrmStateAddOrUpdate(state, nl.XFRM_MSG_UPDSA)
}

func (h *Handle) xfrmStateAddOrUpdate(state *XfrmState, nlProto int) error {

	// A state with spi 0 can't be deleted so don't allow it to be set
	if state.Spi == 0 {
		return fmt.Errorf("Spi must be set when adding xfrm state.")
	}
	req := h.newNetlinkRequest(nlProto, unix.NLM_F_CREATE|unix.NLM_F_EXCL|unix.NLM_F_ACK)

	msg := xfrmUsersaInfoFromXfrmState(state)

	if state.ESN {
		if state.ReplayWindow == 0 {
			return fmt.Errorf("ESN flag set without ReplayWindow")
		}
		msg.Flags |= nl.XFRM_STATE_ESN
		msg.ReplayWindow = 0
	}

	limitsToLft(state.Limits, &msg.Lft)
	req.AddData(msg)

	if state.Auth != nil {
		out := nl.NewRtAttr(nl.XFRMA_ALG_AUTH_TRUNC, writeStateAlgoAuth(state.Auth))
		req.AddData(out)
	}
	if state.Crypt != nil {
		out := nl.NewRtAttr(nl.XFRMA_ALG_CRYPT, writeStateAlgo(state.Crypt))
		req.AddData(out)
	}
	if state.Aead != nil {
		out := nl.NewRtAttr(nl.XFRMA_ALG_AEAD, writeStateAlgoAead(state.Aead))
		req.AddData(out)
	}
	if state.Encap != nil {
		encapData := make([]byte, nl.SizeofXfrmEncapTmpl)
		encap := nl.DeserializeXfrmEncapTmpl(encapData)
		encap.EncapType = uint16(state.Encap.Type)
		encap.EncapSport = nl.Swap16(uint16(state.Encap.SrcPort))
		encap.EncapDport = nl.Swap16(uint16(state.Encap.DstPort))
		encap.EncapOa.FromIP(state.Encap.OriginalAddress)
		out := nl.NewRtAttr(nl.XFRMA_ENCAP, encapData)
		req.AddData(out)
	}
	if state.Mark != nil {
		out := nl.NewRtAttr(nl.XFRMA_MARK, writeMark(state.Mark))
		req.AddData(out)
	}
	if state.ESN {
		out := nl.NewRtAttr(nl.XFRMA_REPLAY_ESN_VAL, writeReplayEsn(state.ReplayWindow))
		req.AddData(out)
	}

	_, err := req.Execute(unix.NETLINK_XFRM, 0)
	return err
}

func (h *Handle) xfrmStateAllocSpi(state *XfrmState) (*XfrmState, error) {
	req := h.newNetlinkRequest(nl.XFRM_MSG_ALLOCSPI,
		unix.NLM_F_CREATE|unix.NLM_F_EXCL|unix.NLM_F_ACK)

	msg := &nl.XfrmUserSpiInfo{}
	msg.XfrmUsersaInfo = *(xfrmUsersaInfoFromXfrmState(state))
	// 1-255 is reserved by IANA for future use
	msg.Min = 0x100
	msg.Max = 0xffffffff
	req.AddData(msg)

	if state.Mark != nil {
		out := nl.NewRtAttr(nl.XFRMA_MARK, writeMark(state.Mark))
		req.AddData(out)
	}

	msgs, err := req.Execute(unix.NETLINK_XFRM, 0)
	if err != nil {
		return nil, err
	}

	s, err := parseXfrmState(msgs[0], FAMILY_ALL)
	if err != nil {
		return nil, err
	}

	return s, err
}

// XfrmStateDel will delete an xfrm state from the system. Note that
// the Algos are ignored when matching the state to delete.
// Equivalent to: `ip xfrm state del $state`
func XfrmStateDel(state *XfrmState) error {
	return pkgHandle.XfrmStateDel(state)
}

// XfrmStateDel will delete an xfrm state from the system. Note that
// the Algos are ignored when matching the state to delete.
// Equivalent to: `ip xfrm state del $state`
func (h *Handle) XfrmStateDel(state *XfrmState) error {
	_, err := h.xfrmStateGetOrDelete(state, nl.XFRM_MSG_DELSA)
	return err
}

// XfrmStateList gets a list of xfrm states in the system.
// Equivalent to: `ip [-4|-6] xfrm state show`.
// The list can be filtered by ip family.
func XfrmStateList(family int) ([]XfrmState, error) {
	return pkgHandle.XfrmStateList(family)
}

// XfrmStateList gets a list of xfrm states in the system.
// Equivalent to: `ip xfrm state show`.
// The list can be filtered by ip family.
func (h *Handle) XfrmStateList(family int) ([]XfrmState, error) {
	req := h.newNetlinkRequest(nl.XFRM_MSG_GETSA, unix.NLM_F_DUMP)

	msgs, err := req.Execute(unix.NETLINK_XFRM, nl.XFRM_MSG_NEWSA)
	if err != nil {
		return nil, err
	}

	var res []XfrmState
	for _, m := range msgs {
		if state, err := parseXfrmState(m, family); err == nil {
			res = append(res, *state)
		} else if err == familyError {
			continue
		} else {
			return nil, err
		}
	}
	return res, nil
}

// XfrmStateGet gets the xfrm state described by the ID, if found.
// Equivalent to: `ip xfrm state get ID [ mark MARK [ mask MASK ] ]`.
// Only the fields which constitue the SA ID must be filled in:
// ID := [ src ADDR ] [ dst ADDR ] [ proto XFRM-PROTO ] [ spi SPI ]
// mark is optional
func XfrmStateGet(state *XfrmState) (*XfrmState, error) {
	return pkgHandle.XfrmStateGet(state)
}

// XfrmStateGet gets the xfrm state described by the ID, if found.
// Equivalent to: `ip xfrm state get ID [ mark MARK [ mask MASK ] ]`.
// Only the fields which constitue the SA ID must be filled in:
// ID := [ src ADDR ] [ dst ADDR ] [ proto XFRM-PROTO ] [ spi SPI ]
// mark is optional
func (h *Handle) XfrmStateGet(state *XfrmState) (*XfrmState, error) {
	return h.xfrmStateGetOrDelete(state, nl.XFRM_MSG_GETSA)
}

func (h *Handle) xfrmStateGetOrDelete(state *XfrmState, nlProto int) (*XfrmState, error) {
	req := h.newNetlinkRequest(nlProto, unix.NLM_F_ACK)

	msg := &nl.XfrmUsersaId{}
	msg.Family = uint16(nl.GetIPFamily(state.Dst))
	msg.Daddr.FromIP(state.Dst)
	msg.Proto = uint8(state.Proto)
	msg.Spi = nl.Swap32(uint32(state.Spi))
	req.AddData(msg)

	if state.Mark != nil {
		out := nl.NewRtAttr(nl.XFRMA_MARK, writeMark(state.Mark))
		req.AddData(out)
	}
	if state.Src != nil {
		out := nl.NewRtAttr(nl.XFRMA_SRCADDR, state.Src.To16())
		req.AddData(out)
	}

	resType := nl.XFRM_MSG_NEWSA
	if nlProto == nl.XFRM_MSG_DELSA {
		resType = 0
	}

	msgs, err := req.Execute(unix.NETLINK_XFRM, uint16(resType))
	if err != nil {
		return nil, err
	}

	if nlProto == nl.XFRM_MSG_DELSA {
		return nil, nil
	}

	s, err := parseXfrmState(msgs[0], FAMILY_ALL)
	if err != nil {
		return nil, err
	}

	return s, nil
}

var familyError = fmt.Errorf("family error")

func xfrmStateFromXfrmUsersaInfo(msg *nl.XfrmUsersaInfo) *XfrmState {
	var state XfrmState

	state.Dst = msg.Id.Daddr.ToIP()
	state.Src = msg.Saddr.ToIP()
	state.Proto = Proto(msg.Id.Proto)
	state.Mode = Mode(msg.Mode)
	state.Spi = int(nl.Swap32(msg.Id.Spi))
	state.Reqid = int(msg.Reqid)
	state.ReplayWindow = int(msg.ReplayWindow)
	lftToLimits(&msg.Lft, &state.Limits)
	curToStats(&msg.Curlft, &msg.Stats, &state.Statistics)

	return &state
}

func parseXfrmState(m []byte, family int) (*XfrmState, error) {
	msg := nl.DeserializeXfrmUsersaInfo(m)

	// This is mainly for the state dump
	if family != FAMILY_ALL && family != int(msg.Family) {
		return nil, familyError
	}

	state := xfrmStateFromXfrmUsersaInfo(msg)

	attrs, err := nl.ParseRouteAttr(m[nl.SizeofXfrmUsersaInfo:])
	if err != nil {
		return nil, err
	}

	for _, attr := range attrs {
		switch attr.Attr.Type {
		case nl.XFRMA_ALG_AUTH, nl.XFRMA_ALG_CRYPT:
			var resAlgo *XfrmStateAlgo
			if attr.Attr.Type == nl.XFRMA_ALG_AUTH {
				if state.Auth == nil {
					state.Auth = new(XfrmStateAlgo)
				}
				resAlgo = state.Auth
			} else {
				state.Crypt = new(XfrmStateAlgo)
				resAlgo = state.Crypt
			}
			algo := nl.DeserializeXfrmAlgo(attr.Value[:])
			(*resAlgo).Name = nl.BytesToString(algo.AlgName[:])
			(*resAlgo).Key = algo.AlgKey
		case nl.XFRMA_ALG_AUTH_TRUNC:
			if state.Auth == nil {
				state.Auth = new(XfrmStateAlgo)
			}
			algo := nl.DeserializeXfrmAlgoAuth(attr.Value[:])
			state.Auth.Name = nl.BytesToString(algo.AlgName[:])
			state.Auth.Key = algo.AlgKey
			state.Auth.TruncateLen = int(algo.AlgTruncLen)
		case nl.XFRMA_ALG_AEAD:
			state.Aead = new(XfrmStateAlgo)
			algo := nl.DeserializeXfrmAlgoAEAD(attr.Value[:])
			state.Aead.Name = nl.BytesToString(algo.AlgName[:])
			state.Aead.Key = algo.AlgKey
			state.Aead.ICVLen = int(algo.AlgICVLen)
		case nl.XFRMA_ENCAP:
			encap := nl.DeserializeXfrmEncapTmpl(attr.Value[:])
			state.Encap = new(XfrmStateEncap)
			state.Encap.Type = EncapType(encap.EncapType)
			state.Encap.SrcPort = int(nl.Swap16(encap.EncapSport))
			state.Encap.DstPort = int(nl.Swap16(encap.EncapDport))
			state.Encap.OriginalAddress = encap.EncapOa.ToIP()
		case nl.XFRMA_MARK:
			mark := nl.DeserializeXfrmMark(attr.Value[:])
			state.Mark = new(XfrmMark)
			state.Mark.Value = mark.Value
			state.Mark.Mask = mark.Mask
		}
	}

	return state, nil
}

// XfrmStateFlush will flush the xfrm state on the system.
// proto = 0 means any transformation protocols
// Equivalent to: `ip xfrm state flush [ proto XFRM-PROTO ]`
func XfrmStateFlush(proto Proto) error {
	return pkgHandle.XfrmStateFlush(proto)
}

// XfrmStateFlush will flush the xfrm state on the system.
// proto = 0 means any transformation protocols
// Equivalent to: `ip xfrm state flush [ proto XFRM-PROTO ]`
func (h *Handle) XfrmStateFlush(proto Proto) error {
	req := h.newNetlinkRequest(nl.XFRM_MSG_FLUSHSA, unix.NLM_F_ACK)

	req.AddData(&nl.XfrmUsersaFlush{Proto: uint8(proto)})

	_, err := req.Execute(unix.NETLINK_XFRM, 0)
	if err != nil {
		return err
	}

	return nil
}

func limitsToLft(lmts XfrmStateLimits, lft *nl.XfrmLifetimeCfg) {
	if lmts.ByteSoft != 0 {
		lft.SoftByteLimit = lmts.ByteSoft
	} else {
		lft.SoftByteLimit = nl.XFRM_INF
	}
	if lmts.ByteHard != 0 {
		lft.HardByteLimit = lmts.ByteHard
	} else {
		lft.HardByteLimit = nl.XFRM_INF
	}
	if lmts.PacketSoft != 0 {
		lft.SoftPacketLimit = lmts.PacketSoft
	} else {
		lft.SoftPacketLimit = nl.XFRM_INF
	}
	if lmts.PacketHard != 0 {
		lft.HardPacketLimit = lmts.PacketHard
	} else {
		lft.HardPacketLimit = nl.XFRM_INF
	}
	lft.SoftAddExpiresSeconds = lmts.TimeSoft
	lft.HardAddExpiresSeconds = lmts.TimeHard
	lft.SoftUseExpiresSeconds = lmts.TimeUseSoft
	lft.HardUseExpiresSeconds = lmts.TimeUseHard
}

func lftToLimits(lft *nl.XfrmLifetimeCfg, lmts *XfrmStateLimits) {
	*lmts = *(*XfrmStateLimits)(unsafe.Pointer(lft))
}

func curToStats(cur *nl.XfrmLifetimeCur, wstats *nl.XfrmStats, stats *XfrmStateStats) {
	stats.Bytes = cur.Bytes
	stats.Packets = cur.Packets
	stats.AddTime = cur.AddTime
	stats.UseTime = cur.UseTime
	stats.ReplayWindow = wstats.ReplayWindow
	stats.Replay = wstats.Replay
	stats.Failed = wstats.IntegrityFailed
}

func xfrmUsersaInfoFromXfrmState(state *XfrmState) *nl.XfrmUsersaInfo {
	msg := &nl.XfrmUsersaInfo{}
	msg.Family = uint16(nl.GetIPFamily(state.Dst))
	msg.Id.Daddr.FromIP(state.Dst)
	msg.Saddr.FromIP(state.Src)
	msg.Id.Proto = uint8(state.Proto)
	msg.Mode = uint8(state.Mode)
	msg.Id.Spi = nl.Swap32(uint32(state.Spi))
	msg.Reqid = uint32(state.Reqid)
	msg.ReplayWindow = uint8(state.ReplayWindow)

	return msg
}
