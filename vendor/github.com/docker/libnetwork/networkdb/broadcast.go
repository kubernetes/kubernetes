package networkdb

import (
	"errors"
	"time"

	"github.com/hashicorp/memberlist"
	"github.com/hashicorp/serf/serf"
)

const broadcastTimeout = 5 * time.Second

type networkEventMessage struct {
	id   string
	node string
	msg  []byte
}

func (m *networkEventMessage) Invalidates(other memberlist.Broadcast) bool {
	otherm := other.(*networkEventMessage)
	return m.id == otherm.id && m.node == otherm.node
}

func (m *networkEventMessage) Message() []byte {
	return m.msg
}

func (m *networkEventMessage) Finished() {
}

func (nDB *NetworkDB) sendNetworkEvent(nid string, event NetworkEvent_Type, ltime serf.LamportTime) error {
	nEvent := NetworkEvent{
		Type:      event,
		LTime:     ltime,
		NodeName:  nDB.config.NodeName,
		NetworkID: nid,
	}

	raw, err := encodeMessage(MessageTypeNetworkEvent, &nEvent)
	if err != nil {
		return err
	}

	nDB.networkBroadcasts.QueueBroadcast(&networkEventMessage{
		msg:  raw,
		id:   nid,
		node: nDB.config.NodeName,
	})
	return nil
}

type nodeEventMessage struct {
	msg    []byte
	notify chan<- struct{}
}

func (m *nodeEventMessage) Invalidates(other memberlist.Broadcast) bool {
	return false
}

func (m *nodeEventMessage) Message() []byte {
	return m.msg
}

func (m *nodeEventMessage) Finished() {
	if m.notify != nil {
		close(m.notify)
	}
}

func (nDB *NetworkDB) sendNodeEvent(event NodeEvent_Type) error {
	nEvent := NodeEvent{
		Type:     event,
		LTime:    nDB.networkClock.Increment(),
		NodeName: nDB.config.NodeName,
	}

	raw, err := encodeMessage(MessageTypeNodeEvent, &nEvent)
	if err != nil {
		return err
	}

	notifyCh := make(chan struct{})
	nDB.nodeBroadcasts.QueueBroadcast(&nodeEventMessage{
		msg:    raw,
		notify: notifyCh,
	})

	nDB.RLock()
	noPeers := len(nDB.nodes) <= 1
	nDB.RUnlock()

	// Message enqueued, do not wait for a send if no peer is present
	if noPeers {
		return nil
	}

	// Wait for the broadcast
	select {
	case <-notifyCh:
	case <-time.After(broadcastTimeout):
		return errors.New("timed out broadcasting node event")
	}

	return nil
}

type tableEventMessage struct {
	id    string
	tname string
	key   string
	msg   []byte
	node  string
}

func (m *tableEventMessage) Invalidates(other memberlist.Broadcast) bool {
	otherm := other.(*tableEventMessage)
	return m.tname == otherm.tname && m.id == otherm.id && m.key == otherm.key
}

func (m *tableEventMessage) Message() []byte {
	return m.msg
}

func (m *tableEventMessage) Finished() {
}

func (nDB *NetworkDB) sendTableEvent(event TableEvent_Type, nid string, tname string, key string, entry *entry) error {
	tEvent := TableEvent{
		Type:      event,
		LTime:     entry.ltime,
		NodeName:  nDB.config.NodeName,
		NetworkID: nid,
		TableName: tname,
		Key:       key,
		Value:     entry.value,
	}

	raw, err := encodeMessage(MessageTypeTableEvent, &tEvent)
	if err != nil {
		return err
	}

	var broadcastQ *memberlist.TransmitLimitedQueue
	nDB.RLock()
	thisNodeNetworks, ok := nDB.networks[nDB.config.NodeName]
	if ok {
		// The network may have been removed
		network, networkOk := thisNodeNetworks[nid]
		if !networkOk {
			nDB.RUnlock()
			return nil
		}

		broadcastQ = network.tableBroadcasts
	}
	nDB.RUnlock()

	// The network may have been removed
	if broadcastQ == nil {
		return nil
	}

	broadcastQ.QueueBroadcast(&tableEventMessage{
		msg:   raw,
		id:    nid,
		tname: tname,
		key:   key,
		node:  nDB.config.NodeName,
	})
	return nil
}
