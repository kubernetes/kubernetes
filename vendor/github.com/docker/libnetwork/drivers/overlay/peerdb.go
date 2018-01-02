package overlay

import (
	"context"
	"fmt"
	"net"
	"sync"
	"syscall"

	"github.com/docker/libnetwork/common"
	"github.com/sirupsen/logrus"
)

const ovPeerTable = "overlay_peer_table"

type peerKey struct {
	peerIP  net.IP
	peerMac net.HardwareAddr
}

type peerEntry struct {
	eid        string
	vtep       net.IP
	peerIPMask net.IPMask
	inSandbox  bool
	isLocal    bool
}

type peerMap struct {
	mp map[string]peerEntry
	sync.Mutex
}

type peerNetworkMap struct {
	mp map[string]*peerMap
	sync.Mutex
}

func (pKey peerKey) String() string {
	return fmt.Sprintf("%s %s", pKey.peerIP, pKey.peerMac)
}

func (pKey *peerKey) Scan(state fmt.ScanState, verb rune) error {
	ipB, err := state.Token(true, nil)
	if err != nil {
		return err
	}

	pKey.peerIP = net.ParseIP(string(ipB))

	macB, err := state.Token(true, nil)
	if err != nil {
		return err
	}

	pKey.peerMac, err = net.ParseMAC(string(macB))
	if err != nil {
		return err
	}

	return nil
}

func (d *driver) peerDbWalk(f func(string, *peerKey, *peerEntry) bool) error {
	d.peerDb.Lock()
	nids := []string{}
	for nid := range d.peerDb.mp {
		nids = append(nids, nid)
	}
	d.peerDb.Unlock()

	for _, nid := range nids {
		d.peerDbNetworkWalk(nid, func(pKey *peerKey, pEntry *peerEntry) bool {
			return f(nid, pKey, pEntry)
		})
	}
	return nil
}

func (d *driver) peerDbNetworkWalk(nid string, f func(*peerKey, *peerEntry) bool) error {
	d.peerDb.Lock()
	pMap, ok := d.peerDb.mp[nid]
	d.peerDb.Unlock()

	if !ok {
		return nil
	}

	mp := map[string]peerEntry{}

	pMap.Lock()
	for pKeyStr, pEntry := range pMap.mp {
		mp[pKeyStr] = pEntry
	}
	pMap.Unlock()

	for pKeyStr, pEntry := range mp {
		var pKey peerKey
		if _, err := fmt.Sscan(pKeyStr, &pKey); err != nil {
			logrus.Warnf("Peer key scan on network %s failed: %v", nid, err)
		}
		if f(&pKey, &pEntry) {
			return nil
		}
	}

	return nil
}

func (d *driver) peerDbSearch(nid string, peerIP net.IP) (net.HardwareAddr, net.IPMask, net.IP, error) {
	var (
		peerMac    net.HardwareAddr
		vtep       net.IP
		peerIPMask net.IPMask
		found      bool
	)

	err := d.peerDbNetworkWalk(nid, func(pKey *peerKey, pEntry *peerEntry) bool {
		if pKey.peerIP.Equal(peerIP) {
			peerMac = pKey.peerMac
			peerIPMask = pEntry.peerIPMask
			vtep = pEntry.vtep
			found = true
			return found
		}

		return found
	})

	if err != nil {
		return nil, nil, nil, fmt.Errorf("peerdb search for peer ip %q failed: %v", peerIP, err)
	}

	if !found {
		return nil, nil, nil, fmt.Errorf("peer ip %q not found in peerdb", peerIP)
	}

	return peerMac, peerIPMask, vtep, nil
}

func (d *driver) peerDbAdd(nid, eid string, peerIP net.IP, peerIPMask net.IPMask,
	peerMac net.HardwareAddr, vtep net.IP, isLocal bool) {

	d.peerDb.Lock()
	pMap, ok := d.peerDb.mp[nid]
	if !ok {
		d.peerDb.mp[nid] = &peerMap{
			mp: make(map[string]peerEntry),
		}

		pMap = d.peerDb.mp[nid]
	}
	d.peerDb.Unlock()

	pKey := peerKey{
		peerIP:  peerIP,
		peerMac: peerMac,
	}

	pEntry := peerEntry{
		eid:        eid,
		vtep:       vtep,
		peerIPMask: peerIPMask,
		isLocal:    isLocal,
	}

	pMap.Lock()
	pMap.mp[pKey.String()] = pEntry
	pMap.Unlock()
}

func (d *driver) peerDbDelete(nid, eid string, peerIP net.IP, peerIPMask net.IPMask,
	peerMac net.HardwareAddr, vtep net.IP) peerEntry {

	d.peerDb.Lock()
	pMap, ok := d.peerDb.mp[nid]
	if !ok {
		d.peerDb.Unlock()
		return peerEntry{}
	}
	d.peerDb.Unlock()

	pKey := peerKey{
		peerIP:  peerIP,
		peerMac: peerMac,
	}

	pMap.Lock()

	pEntry, ok := pMap.mp[pKey.String()]
	if ok {
		// Mismatched endpoint ID(possibly outdated). Do not
		// delete peerdb
		if pEntry.eid != eid {
			pMap.Unlock()
			return pEntry
		}
	}

	delete(pMap.mp, pKey.String())
	pMap.Unlock()

	return pEntry
}

// The overlay uses a lazy initialization approach, this means that when a network is created
// and the driver registered the overlay does not allocate resources till the moment that a
// sandbox is actually created.
// At the moment of this call, that happens when a sandbox is initialized, is possible that
// networkDB has already delivered some events of peers already available on remote nodes,
// these peers are saved into the peerDB and this function is used to properly configure
// the network sandbox with all those peers that got previously notified.
// Note also that this method sends a single message on the channel and the go routine on the
// other side, will atomically loop on the whole table of peers and will program their state
// in one single atomic operation. This is fundamental to guarantee consistency, and avoid that
// new peerAdd or peerDelete gets reordered during the sandbox init.
func (d *driver) initSandboxPeerDB(nid string) {
	d.peerInit(nid)
}

type peerOperationType int32

const (
	peerOperationINIT peerOperationType = iota
	peerOperationADD
	peerOperationDELETE
)

type peerOperation struct {
	opType     peerOperationType
	networkID  string
	endpointID string
	peerIP     net.IP
	peerIPMask net.IPMask
	peerMac    net.HardwareAddr
	vtepIP     net.IP
	l2Miss     bool
	l3Miss     bool
	localPeer  bool
	callerName string
}

func (d *driver) peerOpRoutine(ctx context.Context, ch chan *peerOperation) {
	var err error
	for {
		select {
		case <-ctx.Done():
			return
		case op := <-ch:
			switch op.opType {
			case peerOperationINIT:
				err = d.peerInitOp(op.networkID)
			case peerOperationADD:
				err = d.peerAddOp(op.networkID, op.endpointID, op.peerIP, op.peerIPMask, op.peerMac, op.vtepIP, op.l2Miss, op.l3Miss, true, op.localPeer)
			case peerOperationDELETE:
				err = d.peerDeleteOp(op.networkID, op.endpointID, op.peerIP, op.peerIPMask, op.peerMac, op.vtepIP)
			}
			if err != nil {
				logrus.Warnf("Peer operation failed:%s op:%v", err, op)
			}
		}
	}
}

func (d *driver) peerInit(nid string) {
	callerName := common.CallerName(1)
	d.peerOpCh <- &peerOperation{
		opType:     peerOperationINIT,
		networkID:  nid,
		callerName: callerName,
	}
}

func (d *driver) peerInitOp(nid string) error {
	return d.peerDbNetworkWalk(nid, func(pKey *peerKey, pEntry *peerEntry) bool {
		// Local entries do not need to be added
		if pEntry.isLocal {
			return false
		}

		d.peerAddOp(nid, pEntry.eid, pKey.peerIP, pEntry.peerIPMask, pKey.peerMac, pEntry.vtep, false, false, false, pEntry.isLocal)
		// return false to loop on all entries
		return false
	})
}

func (d *driver) peerAdd(nid, eid string, peerIP net.IP, peerIPMask net.IPMask,
	peerMac net.HardwareAddr, vtep net.IP, l2Miss, l3Miss, localPeer bool) {
	callerName := common.CallerName(1)
	d.peerOpCh <- &peerOperation{
		opType:     peerOperationADD,
		networkID:  nid,
		endpointID: eid,
		peerIP:     peerIP,
		peerIPMask: peerIPMask,
		peerMac:    peerMac,
		vtepIP:     vtep,
		l2Miss:     l2Miss,
		l3Miss:     l3Miss,
		localPeer:  localPeer,
		callerName: callerName,
	}
}

func (d *driver) peerAddOp(nid, eid string, peerIP net.IP, peerIPMask net.IPMask,
	peerMac net.HardwareAddr, vtep net.IP, l2Miss, l3Miss, updateDB, updateOnlyDB bool) error {

	if err := validateID(nid, eid); err != nil {
		return err
	}

	if updateDB {
		d.peerDbAdd(nid, eid, peerIP, peerIPMask, peerMac, vtep, false)
		if updateOnlyDB {
			return nil
		}
	}

	n := d.network(nid)
	if n == nil {
		return nil
	}

	sbox := n.sandbox()
	if sbox == nil {
		// We are hitting this case for all the events that are arriving before that the sandbox
		// is being created. The peer got already added into the database and the sanbox init will
		// call the peerDbUpdateSandbox that will configure all these peers from the database
		return nil
	}

	IP := &net.IPNet{
		IP:   peerIP,
		Mask: peerIPMask,
	}

	s := n.getSubnetforIP(IP)
	if s == nil {
		return fmt.Errorf("couldn't find the subnet %q in network %q", IP.String(), n.id)
	}

	if err := n.obtainVxlanID(s); err != nil {
		return fmt.Errorf("couldn't get vxlan id for %q: %v", s.subnetIP.String(), err)
	}

	if err := n.joinSubnetSandbox(s, false); err != nil {
		return fmt.Errorf("subnet sandbox join failed for %q: %v", s.subnetIP.String(), err)
	}

	if err := d.checkEncryption(nid, vtep, n.vxlanID(s), false, true); err != nil {
		logrus.Warn(err)
	}

	// Add neighbor entry for the peer IP
	if err := sbox.AddNeighbor(peerIP, peerMac, l3Miss, sbox.NeighborOptions().LinkName(s.vxlanName)); err != nil {
		return fmt.Errorf("could not add neighbor entry into the sandbox: %v", err)
	}

	// Add fdb entry to the bridge for the peer mac
	if err := sbox.AddNeighbor(vtep, peerMac, l2Miss, sbox.NeighborOptions().LinkName(s.vxlanName),
		sbox.NeighborOptions().Family(syscall.AF_BRIDGE)); err != nil {
		return fmt.Errorf("could not add fdb entry into the sandbox: %v", err)
	}

	return nil
}

func (d *driver) peerDelete(nid, eid string, peerIP net.IP, peerIPMask net.IPMask,
	peerMac net.HardwareAddr, vtep net.IP) {
	callerName := common.CallerName(1)
	d.peerOpCh <- &peerOperation{
		opType:     peerOperationDELETE,
		networkID:  nid,
		endpointID: eid,
		peerIP:     peerIP,
		peerIPMask: peerIPMask,
		peerMac:    peerMac,
		vtepIP:     vtep,
		callerName: callerName,
	}
}

func (d *driver) peerDeleteOp(nid, eid string, peerIP net.IP, peerIPMask net.IPMask,
	peerMac net.HardwareAddr, vtep net.IP) error {

	if err := validateID(nid, eid); err != nil {
		return err
	}

	pEntry := d.peerDbDelete(nid, eid, peerIP, peerIPMask, peerMac, vtep)

	n := d.network(nid)
	if n == nil {
		return nil
	}

	sbox := n.sandbox()
	if sbox == nil {
		return nil
	}

	// Delete fdb entry to the bridge for the peer mac only if the
	// entry existed in local peerdb. If it is a stale delete
	// request, still call DeleteNeighbor but only to cleanup any
	// leftover sandbox neighbor cache and not actually delete the
	// kernel state.
	if (eid == pEntry.eid && vtep.Equal(pEntry.vtep)) ||
		(eid != pEntry.eid && !vtep.Equal(pEntry.vtep)) {
		if err := sbox.DeleteNeighbor(vtep, peerMac,
			eid == pEntry.eid && vtep.Equal(pEntry.vtep)); err != nil {
			return fmt.Errorf("could not delete fdb entry into the sandbox: %v", err)
		}
	}

	// Delete neighbor entry for the peer IP
	if eid == pEntry.eid {
		if err := sbox.DeleteNeighbor(peerIP, peerMac, true); err != nil {
			return fmt.Errorf("could not delete neighbor entry into the sandbox: %v", err)
		}
	}

	if err := d.checkEncryption(nid, vtep, 0, false, false); err != nil {
		logrus.Warn(err)
	}

	return nil
}

func (d *driver) pushLocalDb() {
	d.peerDbWalk(func(nid string, pKey *peerKey, pEntry *peerEntry) bool {
		if pEntry.isLocal {
			d.pushLocalEndpointEvent("join", nid, pEntry.eid)
		}
		return false
	})
}

func (d *driver) peerDBUpdateSelf() {
	d.peerDbWalk(func(nid string, pkey *peerKey, pEntry *peerEntry) bool {
		if pEntry.isLocal {
			pEntry.vtep = net.ParseIP(d.advertiseAddress)
		}
		return false
	})
}
