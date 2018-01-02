package overlay

import (
	"fmt"
	"net"
	"strings"
	"time"

	"github.com/hashicorp/serf/serf"
	"github.com/sirupsen/logrus"
)

type ovNotify struct {
	action string
	ep     *endpoint
	nw     *network
}

type logWriter struct{}

func (l *logWriter) Write(p []byte) (int, error) {
	str := string(p)

	switch {
	case strings.Contains(str, "[WARN]"):
		logrus.Warn(str)
	case strings.Contains(str, "[DEBUG]"):
		logrus.Debug(str)
	case strings.Contains(str, "[INFO]"):
		logrus.Info(str)
	case strings.Contains(str, "[ERR]"):
		logrus.Error(str)
	}

	return len(p), nil
}

func (d *driver) serfInit() error {
	var err error

	config := serf.DefaultConfig()
	config.Init()
	config.MemberlistConfig.BindAddr = d.advertiseAddress

	d.eventCh = make(chan serf.Event, 4)
	config.EventCh = d.eventCh
	config.UserCoalescePeriod = 1 * time.Second
	config.UserQuiescentPeriod = 50 * time.Millisecond

	config.LogOutput = &logWriter{}
	config.MemberlistConfig.LogOutput = config.LogOutput

	s, err := serf.Create(config)
	if err != nil {
		return fmt.Errorf("failed to create cluster node: %v", err)
	}
	defer func() {
		if err != nil {
			s.Shutdown()
		}
	}()

	d.serfInstance = s

	d.notifyCh = make(chan ovNotify)
	d.exitCh = make(chan chan struct{})

	go d.startSerfLoop(d.eventCh, d.notifyCh, d.exitCh)
	return nil
}

func (d *driver) serfJoin(neighIP string) error {
	if neighIP == "" {
		return fmt.Errorf("no neighbor to join")
	}
	if _, err := d.serfInstance.Join([]string{neighIP}, false); err != nil {
		return fmt.Errorf("Failed to join the cluster at neigh IP %s: %v",
			neighIP, err)
	}
	return nil
}

func (d *driver) notifyEvent(event ovNotify) {
	ep := event.ep

	ePayload := fmt.Sprintf("%s %s %s %s", event.action, ep.addr.IP.String(),
		net.IP(ep.addr.Mask).String(), ep.mac.String())
	eName := fmt.Sprintf("jl %s %s %s", d.serfInstance.LocalMember().Addr.String(),
		event.nw.id, ep.id)

	if err := d.serfInstance.UserEvent(eName, []byte(ePayload), true); err != nil {
		logrus.Errorf("Sending user event failed: %v\n", err)
	}
}

func (d *driver) processEvent(u serf.UserEvent) {
	logrus.Debugf("Received user event name:%s, payload:%s\n", u.Name,
		string(u.Payload))

	var dummy, action, vtepStr, nid, eid, ipStr, maskStr, macStr string
	if _, err := fmt.Sscan(u.Name, &dummy, &vtepStr, &nid, &eid); err != nil {
		fmt.Printf("Failed to scan name string: %v\n", err)
	}

	if _, err := fmt.Sscan(string(u.Payload), &action,
		&ipStr, &maskStr, &macStr); err != nil {
		fmt.Printf("Failed to scan value string: %v\n", err)
	}

	logrus.Debugf("Parsed data = %s/%s/%s/%s/%s/%s\n", nid, eid, vtepStr, ipStr, maskStr, macStr)

	mac, err := net.ParseMAC(macStr)
	if err != nil {
		logrus.Errorf("Failed to parse mac: %v\n", err)
	}

	if d.serfInstance.LocalMember().Addr.String() == vtepStr {
		return
	}

	switch action {
	case "join":
		if err := d.peerAdd(nid, eid, net.ParseIP(ipStr), net.IPMask(net.ParseIP(maskStr).To4()), mac,
			net.ParseIP(vtepStr), true); err != nil {
			logrus.Errorf("Peer add failed in the driver: %v\n", err)
		}
	case "leave":
		if err := d.peerDelete(nid, eid, net.ParseIP(ipStr), net.IPMask(net.ParseIP(maskStr).To4()), mac,
			net.ParseIP(vtepStr), true); err != nil {
			logrus.Errorf("Peer delete failed in the driver: %v\n", err)
		}
	}
}

func (d *driver) processQuery(q *serf.Query) {
	logrus.Debugf("Received query name:%s, payload:%s\n", q.Name,
		string(q.Payload))

	var nid, ipStr string
	if _, err := fmt.Sscan(string(q.Payload), &nid, &ipStr); err != nil {
		fmt.Printf("Failed to scan query payload string: %v\n", err)
	}

	peerMac, peerIPMask, vtep, err := d.peerDbSearch(nid, net.ParseIP(ipStr))
	if err != nil {
		return
	}

	q.Respond([]byte(fmt.Sprintf("%s %s %s", peerMac.String(), net.IP(peerIPMask).String(), vtep.String())))
}

func (d *driver) resolvePeer(nid string, peerIP net.IP) (net.HardwareAddr, net.IPMask, net.IP, error) {
	if d.serfInstance == nil {
		return nil, nil, nil, fmt.Errorf("could not resolve peer: serf instance not initialized")
	}

	qPayload := fmt.Sprintf("%s %s", string(nid), peerIP.String())
	resp, err := d.serfInstance.Query("peerlookup", []byte(qPayload), nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("resolving peer by querying the cluster failed: %v", err)
	}

	respCh := resp.ResponseCh()
	select {
	case r := <-respCh:
		var macStr, maskStr, vtepStr string
		if _, err := fmt.Sscan(string(r.Payload), &macStr, &maskStr, &vtepStr); err != nil {
			return nil, nil, nil, fmt.Errorf("bad response %q for the resolve query: %v", string(r.Payload), err)
		}

		mac, err := net.ParseMAC(macStr)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to parse mac: %v", err)
		}

		return mac, net.IPMask(net.ParseIP(maskStr).To4()), net.ParseIP(vtepStr), nil

	case <-time.After(time.Second):
		return nil, nil, nil, fmt.Errorf("timed out resolving peer by querying the cluster")
	}
}

func (d *driver) startSerfLoop(eventCh chan serf.Event, notifyCh chan ovNotify,
	exitCh chan chan struct{}) {

	for {
		select {
		case notify, ok := <-notifyCh:
			if !ok {
				break
			}

			d.notifyEvent(notify)
		case ch, ok := <-exitCh:
			if !ok {
				break
			}

			if err := d.serfInstance.Leave(); err != nil {
				logrus.Errorf("failed leaving the cluster: %v\n", err)
			}

			d.serfInstance.Shutdown()
			close(ch)
			return
		case e, ok := <-eventCh:
			if !ok {
				break
			}

			if e.EventType() == serf.EventQuery {
				d.processQuery(e.(*serf.Query))
				break
			}

			u, ok := e.(serf.UserEvent)
			if !ok {
				break
			}
			d.processEvent(u)
		}
	}
}

func (d *driver) isSerfAlive() bool {
	d.Lock()
	serfInstance := d.serfInstance
	d.Unlock()
	if serfInstance == nil || serfInstance.State() != serf.SerfAlive {
		return false
	}
	return true
}
