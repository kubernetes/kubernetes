package stressClient

import (
	"sync"
)

// The Tracer carrys tags and a waitgroup from the statements through the package life cycle
type Tracer struct {
	Tags map[string]string

	sync.WaitGroup
}

// NewTracer returns a Tracer with tags attached
func NewTracer(tags map[string]string) *Tracer {
	return &Tracer{
		Tags: tags,
	}
}
