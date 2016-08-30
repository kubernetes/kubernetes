package serf

import (
	"github.com/hashicorp/memberlist"
)

type eventDelegate struct {
	serf *Serf
}

func (e *eventDelegate) NotifyJoin(n *memberlist.Node) {
	e.serf.handleNodeJoin(n)
}

func (e *eventDelegate) NotifyLeave(n *memberlist.Node) {
	e.serf.handleNodeLeave(n)
}

func (e *eventDelegate) NotifyUpdate(n *memberlist.Node) {
	e.serf.handleNodeUpdate(n)
}
