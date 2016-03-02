package callback

import (
	"github.com/mesos/mesos-go/upid"
)

type Interprocess struct {
	client upid.UPID
	server upid.UPID
}

func NewInterprocess() *Interprocess {
	return &Interprocess{}
}

func (cb *Interprocess) Client() upid.UPID {
	return cb.client
}

func (cb *Interprocess) Server() upid.UPID {
	return cb.server
}

func (cb *Interprocess) Set(server, client upid.UPID) {
	cb.server = server
	cb.client = client
}
