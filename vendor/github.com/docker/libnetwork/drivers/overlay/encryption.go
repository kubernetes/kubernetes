package overlay

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"hash/fnv"
	"net"
	"sync"
	"syscall"

	"strconv"

	"github.com/docker/libnetwork/iptables"
	"github.com/docker/libnetwork/ns"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
	"github.com/vishvananda/netlink"
)

const (
	r            = 0xD0C4E3
	timeout      = 30
	pktExpansion = 26 // SPI(4) + SeqN(4) + IV(8) + PadLength(1) + NextHeader(1) + ICV(8)
)

const (
	forward = iota + 1
	reverse
	bidir
)

var spMark = netlink.XfrmMark{Value: uint32(r), Mask: 0xffffffff}

type key struct {
	value []byte
	tag   uint32
}

func (k *key) String() string {
	if k != nil {
		return fmt.Sprintf("(key: %s, tag: 0x%x)", hex.EncodeToString(k.value)[0:5], k.tag)
	}
	return ""
}

type spi struct {
	forward int
	reverse int
}

func (s *spi) String() string {
	return fmt.Sprintf("SPI(FWD: 0x%x, REV: 0x%x)", uint32(s.forward), uint32(s.reverse))
}

type encrMap struct {
	nodes map[string][]*spi
	sync.Mutex
}

func (e *encrMap) String() string {
	e.Lock()
	defer e.Unlock()
	b := new(bytes.Buffer)
	for k, v := range e.nodes {
		b.WriteString("\n")
		b.WriteString(k)
		b.WriteString(":")
		b.WriteString("[")
		for _, s := range v {
			b.WriteString(s.String())
			b.WriteString(",")
		}
		b.WriteString("]")

	}
	return b.String()
}

func (d *driver) checkEncryption(nid string, rIP net.IP, vxlanID uint32, isLocal, add bool) error {
	logrus.Debugf("checkEncryption(%s, %v, %d, %t)", nid[0:7], rIP, vxlanID, isLocal)

	n := d.network(nid)
	if n == nil || !n.secure {
		return nil
	}

	if len(d.keys) == 0 {
		return types.ForbiddenErrorf("encryption key is not present")
	}

	lIP := net.ParseIP(d.bindAddress)
	aIP := net.ParseIP(d.advertiseAddress)
	nodes := map[string]net.IP{}

	switch {
	case isLocal:
		if err := d.peerDbNetworkWalk(nid, func(pKey *peerKey, pEntry *peerEntry) bool {
			if !aIP.Equal(pEntry.vtep) {
				nodes[pEntry.vtep.String()] = pEntry.vtep
			}
			return false
		}); err != nil {
			logrus.Warnf("Failed to retrieve list of participating nodes in overlay network %s: %v", nid[0:5], err)
		}
	default:
		if len(d.network(nid).endpoints) > 0 {
			nodes[rIP.String()] = rIP
		}
	}

	logrus.Debugf("List of nodes: %s", nodes)

	if add {
		for _, rIP := range nodes {
			if err := setupEncryption(lIP, aIP, rIP, vxlanID, d.secMap, d.keys); err != nil {
				logrus.Warnf("Failed to program network encryption between %s and %s: %v", lIP, rIP, err)
			}
		}
	} else {
		if len(nodes) == 0 {
			if err := removeEncryption(lIP, rIP, d.secMap); err != nil {
				logrus.Warnf("Failed to remove network encryption between %s and %s: %v", lIP, rIP, err)
			}
		}
	}

	return nil
}

func setupEncryption(localIP, advIP, remoteIP net.IP, vni uint32, em *encrMap, keys []*key) error {
	logrus.Debugf("Programming encryption for vxlan %d between %s and %s", vni, localIP, remoteIP)
	rIPs := remoteIP.String()

	indices := make([]*spi, 0, len(keys))

	err := programMangle(vni, true)
	if err != nil {
		logrus.Warn(err)
	}

	err = programInput(vni, true)
	if err != nil {
		logrus.Warn(err)
	}

	for i, k := range keys {
		spis := &spi{buildSPI(advIP, remoteIP, k.tag), buildSPI(remoteIP, advIP, k.tag)}
		dir := reverse
		if i == 0 {
			dir = bidir
		}
		fSA, rSA, err := programSA(localIP, remoteIP, spis, k, dir, true)
		if err != nil {
			logrus.Warn(err)
		}
		indices = append(indices, spis)
		if i != 0 {
			continue
		}
		err = programSP(fSA, rSA, true)
		if err != nil {
			logrus.Warn(err)
		}
	}

	em.Lock()
	em.nodes[rIPs] = indices
	em.Unlock()

	return nil
}

func removeEncryption(localIP, remoteIP net.IP, em *encrMap) error {
	em.Lock()
	indices, ok := em.nodes[remoteIP.String()]
	em.Unlock()
	if !ok {
		return nil
	}
	for i, idxs := range indices {
		dir := reverse
		if i == 0 {
			dir = bidir
		}
		fSA, rSA, err := programSA(localIP, remoteIP, idxs, nil, dir, false)
		if err != nil {
			logrus.Warn(err)
		}
		if i != 0 {
			continue
		}
		err = programSP(fSA, rSA, false)
		if err != nil {
			logrus.Warn(err)
		}
	}
	return nil
}

func programMangle(vni uint32, add bool) (err error) {
	var (
		p      = strconv.FormatUint(uint64(vxlanPort), 10)
		c      = fmt.Sprintf("0>>22&0x3C@12&0xFFFFFF00=%d", int(vni)<<8)
		m      = strconv.FormatUint(uint64(r), 10)
		chain  = "OUTPUT"
		rule   = []string{"-p", "udp", "--dport", p, "-m", "u32", "--u32", c, "-j", "MARK", "--set-mark", m}
		a      = "-A"
		action = "install"
	)

	if add == iptables.Exists(iptables.Mangle, chain, rule...) {
		return
	}

	if !add {
		a = "-D"
		action = "remove"
	}

	if err = iptables.RawCombinedOutput(append([]string{"-t", string(iptables.Mangle), a, chain}, rule...)...); err != nil {
		logrus.Warnf("could not %s mangle rule: %v", action, err)
	}

	return
}

func programInput(vni uint32, add bool) (err error) {
	var (
		port       = strconv.FormatUint(uint64(vxlanPort), 10)
		vniMatch   = fmt.Sprintf("0>>22&0x3C@12&0xFFFFFF00=%d", int(vni)<<8)
		plainVxlan = []string{"-p", "udp", "--dport", port, "-m", "u32", "--u32", vniMatch, "-j"}
		ipsecVxlan = append([]string{"-m", "policy", "--dir", "in", "--pol", "ipsec"}, plainVxlan...)
		block      = append(plainVxlan, "DROP")
		accept     = append(ipsecVxlan, "ACCEPT")
		chain      = "INPUT"
		action     = iptables.Append
		msg        = "add"
	)

	if !add {
		action = iptables.Delete
		msg = "remove"
	}

	if err := iptables.ProgramRule(iptables.Filter, chain, action, accept); err != nil {
		logrus.Errorf("could not %s input rule: %v. Please do it manually.", msg, err)
	}

	if err := iptables.ProgramRule(iptables.Filter, chain, action, block); err != nil {
		logrus.Errorf("could not %s input rule: %v. Please do it manually.", msg, err)
	}

	return
}

func programSA(localIP, remoteIP net.IP, spi *spi, k *key, dir int, add bool) (fSA *netlink.XfrmState, rSA *netlink.XfrmState, err error) {
	var (
		action      = "Removing"
		xfrmProgram = ns.NlHandle().XfrmStateDel
	)

	if add {
		action = "Adding"
		xfrmProgram = ns.NlHandle().XfrmStateAdd
	}

	if dir&reverse > 0 {
		rSA = &netlink.XfrmState{
			Src:   remoteIP,
			Dst:   localIP,
			Proto: netlink.XFRM_PROTO_ESP,
			Spi:   spi.reverse,
			Mode:  netlink.XFRM_MODE_TRANSPORT,
			Reqid: r,
		}
		if add {
			rSA.Aead = buildAeadAlgo(k, spi.reverse)
		}

		exists, err := saExists(rSA)
		if err != nil {
			exists = !add
		}

		if add != exists {
			logrus.Debugf("%s: rSA{%s}", action, rSA)
			if err := xfrmProgram(rSA); err != nil {
				logrus.Warnf("Failed %s rSA{%s}: %v", action, rSA, err)
			}
		}
	}

	if dir&forward > 0 {
		fSA = &netlink.XfrmState{
			Src:   localIP,
			Dst:   remoteIP,
			Proto: netlink.XFRM_PROTO_ESP,
			Spi:   spi.forward,
			Mode:  netlink.XFRM_MODE_TRANSPORT,
			Reqid: r,
		}
		if add {
			fSA.Aead = buildAeadAlgo(k, spi.forward)
		}

		exists, err := saExists(fSA)
		if err != nil {
			exists = !add
		}

		if add != exists {
			logrus.Debugf("%s fSA{%s}", action, fSA)
			if err := xfrmProgram(fSA); err != nil {
				logrus.Warnf("Failed %s fSA{%s}: %v.", action, fSA, err)
			}
		}
	}

	return
}

func programSP(fSA *netlink.XfrmState, rSA *netlink.XfrmState, add bool) error {
	action := "Removing"
	xfrmProgram := ns.NlHandle().XfrmPolicyDel
	if add {
		action = "Adding"
		xfrmProgram = ns.NlHandle().XfrmPolicyAdd
	}

	// Create a congruent cidr
	s := types.GetMinimalIP(fSA.Src)
	d := types.GetMinimalIP(fSA.Dst)
	fullMask := net.CIDRMask(8*len(s), 8*len(s))

	fPol := &netlink.XfrmPolicy{
		Src:     &net.IPNet{IP: s, Mask: fullMask},
		Dst:     &net.IPNet{IP: d, Mask: fullMask},
		Dir:     netlink.XFRM_DIR_OUT,
		Proto:   17,
		DstPort: 4789,
		Mark:    &spMark,
		Tmpls: []netlink.XfrmPolicyTmpl{
			{
				Src:   fSA.Src,
				Dst:   fSA.Dst,
				Proto: netlink.XFRM_PROTO_ESP,
				Mode:  netlink.XFRM_MODE_TRANSPORT,
				Spi:   fSA.Spi,
				Reqid: r,
			},
		},
	}

	exists, err := spExists(fPol)
	if err != nil {
		exists = !add
	}

	if add != exists {
		logrus.Debugf("%s fSP{%s}", action, fPol)
		if err := xfrmProgram(fPol); err != nil {
			logrus.Warnf("%s fSP{%s}: %v", action, fPol, err)
		}
	}

	return nil
}

func saExists(sa *netlink.XfrmState) (bool, error) {
	_, err := ns.NlHandle().XfrmStateGet(sa)
	switch err {
	case nil:
		return true, nil
	case syscall.ESRCH:
		return false, nil
	default:
		err = fmt.Errorf("Error while checking for SA existence: %v", err)
		logrus.Warn(err)
		return false, err
	}
}

func spExists(sp *netlink.XfrmPolicy) (bool, error) {
	_, err := ns.NlHandle().XfrmPolicyGet(sp)
	switch err {
	case nil:
		return true, nil
	case syscall.ENOENT:
		return false, nil
	default:
		err = fmt.Errorf("Error while checking for SP existence: %v", err)
		logrus.Warn(err)
		return false, err
	}
}

func buildSPI(src, dst net.IP, st uint32) int {
	b := make([]byte, 4)
	binary.BigEndian.PutUint32(b, st)
	h := fnv.New32a()
	h.Write(src)
	h.Write(b)
	h.Write(dst)
	return int(binary.BigEndian.Uint32(h.Sum(nil)))
}

func buildAeadAlgo(k *key, s int) *netlink.XfrmStateAlgo {
	salt := make([]byte, 4)
	binary.BigEndian.PutUint32(salt, uint32(s))
	return &netlink.XfrmStateAlgo{
		Name:   "rfc4106(gcm(aes))",
		Key:    append(k.value, salt...),
		ICVLen: 64,
	}
}

func (d *driver) secMapWalk(f func(string, []*spi) ([]*spi, bool)) error {
	d.secMap.Lock()
	for node, indices := range d.secMap.nodes {
		idxs, stop := f(node, indices)
		if idxs != nil {
			d.secMap.nodes[node] = idxs
		}
		if stop {
			break
		}
	}
	d.secMap.Unlock()
	return nil
}

func (d *driver) setKeys(keys []*key) error {
	// Remove any stale policy, state
	clearEncryptionStates()
	// Accept the encryption keys and clear any stale encryption map
	d.Lock()
	d.keys = keys
	d.secMap = &encrMap{nodes: map[string][]*spi{}}
	d.Unlock()
	logrus.Debugf("Initial encryption keys: %v", d.keys)
	return nil
}

// updateKeys allows to add a new key and/or change the primary key and/or prune an existing key
// The primary key is the key used in transmission and will go in first position in the list.
func (d *driver) updateKeys(newKey, primary, pruneKey *key) error {
	logrus.Debugf("Updating Keys. New: %v, Primary: %v, Pruned: %v", newKey, primary, pruneKey)

	logrus.Debugf("Current: %v", d.keys)

	var (
		newIdx = -1
		priIdx = -1
		delIdx = -1
		lIP    = net.ParseIP(d.bindAddress)
		aIP    = net.ParseIP(d.advertiseAddress)
	)

	d.Lock()
	// add new
	if newKey != nil {
		d.keys = append(d.keys, newKey)
		newIdx += len(d.keys)
	}
	for i, k := range d.keys {
		if primary != nil && k.tag == primary.tag {
			priIdx = i
		}
		if pruneKey != nil && k.tag == pruneKey.tag {
			delIdx = i
		}
	}
	d.Unlock()

	if (newKey != nil && newIdx == -1) ||
		(primary != nil && priIdx == -1) ||
		(pruneKey != nil && delIdx == -1) {
		return types.BadRequestErrorf("cannot find proper key indices while processing key update:"+
			"(newIdx,priIdx,delIdx):(%d, %d, %d)", newIdx, priIdx, delIdx)
	}

	d.secMapWalk(func(rIPs string, spis []*spi) ([]*spi, bool) {
		rIP := net.ParseIP(rIPs)
		return updateNodeKey(lIP, aIP, rIP, spis, d.keys, newIdx, priIdx, delIdx), false
	})

	d.Lock()
	// swap primary
	if priIdx != -1 {
		swp := d.keys[0]
		d.keys[0] = d.keys[priIdx]
		d.keys[priIdx] = swp
	}
	// prune
	if delIdx != -1 {
		if delIdx == 0 {
			delIdx = priIdx
		}
		d.keys = append(d.keys[:delIdx], d.keys[delIdx+1:]...)
	}
	d.Unlock()

	logrus.Debugf("Updated: %v", d.keys)

	return nil
}

/********************************************************
 * Steady state: rSA0, rSA1, rSA2, fSA1, fSP1
 * Rotation --> -rSA0, +rSA3, +fSA2, +fSP2/-fSP1, -fSA1
 * Steady state: rSA1, rSA2, rSA3, fSA2, fSP2
 *********************************************************/

// Spis and keys are sorted in such away the one in position 0 is the primary
func updateNodeKey(lIP, aIP, rIP net.IP, idxs []*spi, curKeys []*key, newIdx, priIdx, delIdx int) []*spi {
	logrus.Debugf("Updating keys for node: %s (%d,%d,%d)", rIP, newIdx, priIdx, delIdx)

	spis := idxs
	logrus.Debugf("Current: %v", spis)

	// add new
	if newIdx != -1 {
		spis = append(spis, &spi{
			forward: buildSPI(aIP, rIP, curKeys[newIdx].tag),
			reverse: buildSPI(rIP, aIP, curKeys[newIdx].tag),
		})
	}

	if delIdx != -1 {
		// -rSA0
		programSA(lIP, rIP, spis[delIdx], nil, reverse, false)
	}

	if newIdx > -1 {
		// +rSA2
		programSA(lIP, rIP, spis[newIdx], curKeys[newIdx], reverse, true)
	}

	if priIdx > 0 {
		// +fSA2
		fSA2, _, _ := programSA(lIP, rIP, spis[priIdx], curKeys[priIdx], forward, true)

		// +fSP2, -fSP1
		s := types.GetMinimalIP(fSA2.Src)
		d := types.GetMinimalIP(fSA2.Dst)
		fullMask := net.CIDRMask(8*len(s), 8*len(s))

		fSP1 := &netlink.XfrmPolicy{
			Src:     &net.IPNet{IP: s, Mask: fullMask},
			Dst:     &net.IPNet{IP: d, Mask: fullMask},
			Dir:     netlink.XFRM_DIR_OUT,
			Proto:   17,
			DstPort: 4789,
			Mark:    &spMark,
			Tmpls: []netlink.XfrmPolicyTmpl{
				{
					Src:   fSA2.Src,
					Dst:   fSA2.Dst,
					Proto: netlink.XFRM_PROTO_ESP,
					Mode:  netlink.XFRM_MODE_TRANSPORT,
					Spi:   fSA2.Spi,
					Reqid: r,
				},
			},
		}
		logrus.Debugf("Updating fSP{%s}", fSP1)
		if err := ns.NlHandle().XfrmPolicyUpdate(fSP1); err != nil {
			logrus.Warnf("Failed to update fSP{%s}: %v", fSP1, err)
		}

		// -fSA1
		programSA(lIP, rIP, spis[0], nil, forward, false)
	}

	// swap
	if priIdx > 0 {
		swp := spis[0]
		spis[0] = spis[priIdx]
		spis[priIdx] = swp
	}
	// prune
	if delIdx != -1 {
		if delIdx == 0 {
			delIdx = priIdx
		}
		spis = append(spis[:delIdx], spis[delIdx+1:]...)
	}

	logrus.Debugf("Updated: %v", spis)

	return spis
}

func (n *network) maxMTU() int {
	mtu := 1500
	if n.mtu != 0 {
		mtu = n.mtu
	}
	mtu -= vxlanEncap
	if n.secure {
		// In case of encryption account for the
		// esp packet espansion and padding
		mtu -= pktExpansion
		mtu -= (mtu % 4)
	}
	return mtu
}

func clearEncryptionStates() {
	nlh := ns.NlHandle()
	spList, err := nlh.XfrmPolicyList(netlink.FAMILY_ALL)
	if err != nil {
		logrus.Warnf("Failed to retrieve SP list for cleanup: %v", err)
	}
	saList, err := nlh.XfrmStateList(netlink.FAMILY_ALL)
	if err != nil {
		logrus.Warnf("Failed to retrieve SA list for cleanup: %v", err)
	}
	for _, sp := range spList {
		if sp.Mark != nil && sp.Mark.Value == spMark.Value {
			if err := nlh.XfrmPolicyDel(&sp); err != nil {
				logrus.Warnf("Failed to delete stale SP %s: %v", sp, err)
				continue
			}
			logrus.Debugf("Removed stale SP: %s", sp)
		}
	}
	for _, sa := range saList {
		if sa.Reqid == r {
			if err := nlh.XfrmStateDel(&sa); err != nil {
				logrus.Warnf("Failed to delete stale SA %s: %v", sa, err)
				continue
			}
			logrus.Debugf("Removed stale SA: %s", sa)
		}
	}
}
