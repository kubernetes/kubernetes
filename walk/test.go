package main

import (
	"github.com/davecgh/go-spew/spew"
	apiutil "k8s.io/kubernetes/pkg/api/util"
	"k8s.io/kubernetes/pkg/api/v1"
)

func main() {
	ep1 := v1.EndpointPort{}
	ep2 := ep1
	apiutil.ApplyDefaults(&ep2)
	spew.Dump(ep1)
	spew.Dump(ep2)

	p1 := v1.Pod{}
	p2 := p1
	apiutil.ApplyDefaults(&p2)
	spew.Dump(p1)
	spew.Dump(p2)
}
