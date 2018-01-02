package overlay

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"hash/fnv"
	"net"
	"sync"

	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
)

const (
	mark         = uint32(0xD0C4E3)
	timeout      = 30
	pktExpansion = 26 // SPI(4) + SeqN(4) + IV(8) + PadLength(1) + NextHeader(1) + ICV(8)
)

const (
	forward = iota + 1
	reverse
	bidir
)

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

	em.Lock()
	em.nodes[rIPs] = indices
	em.Unlock()

	return nil
}

func removeEncryption(localIP, remoteIP net.IP, em *encrMap) error {
	return nil
}

func programMangle(vni uint32, add bool) (err error) {
	return
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
	if d.keys != nil {
		return types.ForbiddenErrorf("initial keys are already present")
	}
	d.keys = keys
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
		err := types.BadRequestErrorf("cannot find proper key indices while processing key update:"+
			"(newIdx,priIdx,delIdx):(%d, %d, %d)", newIdx, priIdx, delIdx)
		logrus.Warn(err)
		return err
	}

	d.secMapWalk(func(rIPs string, spis []*spi) ([]*spi, bool) {
		rIP := net.ParseIP(rIPs)
		return updateNodeKey(lIP, rIP, spis, d.keys, newIdx, priIdx, delIdx), false
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
func updateNodeKey(lIP, rIP net.IP, idxs []*spi, curKeys []*key, newIdx, priIdx, delIdx int) []*spi {
	logrus.Debugf("Updating keys for node: %s (%d,%d,%d)", rIP, newIdx, priIdx, delIdx)
	return nil
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
