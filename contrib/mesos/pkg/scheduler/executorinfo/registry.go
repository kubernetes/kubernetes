/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package executorinfo

import (
	"fmt"
	"strings"
	"sync"

	"github.com/gogo/protobuf/proto"
	"github.com/mesos/mesos-go/mesosproto"
	"k8s.io/kubernetes/contrib/mesos/pkg/node"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
)

// Registry is the interface that provides methods for interacting
// with a registry of ExecutorInfo objects
//
// Get looks up an ExecutorInfo object for the given hostname
//
// New returns an ExecutorInfo object based on a given hostname and resources
//
// Invalidate invalidates the given hostname from this registry.
// Note that a subsequent Get may recover the executor info.
type Registry interface {
	New(hostname string, resources []*mesosproto.Resource) *mesosproto.ExecutorInfo
	Get(hostname string) (*mesosproto.ExecutorInfo, error)
	Invalidate(hostname string)
}

// registry implements a map-based in-memory ExecutorInfo registry
type registry struct {
	cache *Cache
	mu    sync.RWMutex // protects fields above

	lookupNode node.LookupFunc
	prototype  *mesosproto.ExecutorInfo
}

// NewRegistry returns a new executorinfo registry.
// The given prototype is being used for properties other than resources.
func NewRegistry(
	lookupNode node.LookupFunc,
	prototype *mesosproto.ExecutorInfo,
	cache *Cache,
) (Registry, error) {
	if prototype == nil {
		return nil, fmt.Errorf("no prototype given")
	}

	if lookupNode == nil {
		return nil, fmt.Errorf("no lookupNode given")
	}

	if cache == nil {
		return nil, fmt.Errorf("no cache given")
	}

	return &registry{
		cache:      cache,
		lookupNode: lookupNode,
		prototype:  prototype,
	}, nil
}

// New creates a customized ExecutorInfo for a host
//
// Note: New modifies Command.Arguments and Resources and intentionally
// does not update the executor id (although that originally depended on the
// command arguments and the resources). But as the hostname is constant for a
// given host, and the resources are compatible by the registry logic here this
// will not weaken our litmus test comparing the prototype ExecutorId with the
// id of running executors when an offer comes in.
func (r *registry) New(
	hostname string,
	resources []*mesosproto.Resource,
) *mesosproto.ExecutorInfo {
	e := proto.Clone(r.prototype).(*mesosproto.ExecutorInfo)
	e.Resources = resources
	setCommandArgument(e, "--hostname-override", hostname)

	r.mu.Lock()
	defer r.mu.Unlock()

	cached, ok := r.cache.Get(hostname)
	if ok {
		return cached
	}

	r.cache.Add(hostname, e)
	return e
}

func (r *registry) Get(hostname string) (*mesosproto.ExecutorInfo, error) {
	// first try to read from cached items
	r.mu.RLock()
	info, ok := r.cache.Get(hostname)
	r.mu.RUnlock()

	if ok {
		return info, nil
	}

	result, err := r.resourcesFromNode(hostname)
	if err != nil {
		// master claims there is an executor with id, we cannot find any meta info
		// => no way to recover this node
		return nil, fmt.Errorf(
			"failed to recover executor info for node %q, error: %v",
			hostname, err,
		)
	}

	return r.New(hostname, result), nil
}

func (r *registry) Invalidate(hostname string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.cache.Remove(hostname)
}

// resourcesFromNode looks up ExecutorInfo resources for the given hostname and executorinfo ID
// or returns an error in case of failure.
func (r *registry) resourcesFromNode(hostname string) ([]*mesosproto.Resource, error) {
	n := r.lookupNode(hostname)
	if n == nil {
		return nil, fmt.Errorf("hostname %q not found", hostname)
	}

	encoded, ok := n.Annotations[meta.ExecutorResourcesKey]
	if !ok {
		return nil, fmt.Errorf(
			"no %q annotation found in hostname %q",
			meta.ExecutorResourcesKey, hostname,
		)
	}

	return DecodeResources(strings.NewReader(encoded))
}

// setCommandArgument sets the given flag to the given value
// in the command arguments of the given executoringfo.
func setCommandArgument(ei *mesosproto.ExecutorInfo, flag, value string) {
	if ei.Command == nil {
		return
	}

	argv := ei.Command.Arguments
	overwrite := false

	for i, arg := range argv {
		if strings.HasPrefix(arg, flag+"=") {
			overwrite = true
			argv[i] = flag + "=" + value
			break
		}
	}

	if !overwrite {
		ei.Command.Arguments = append(argv, flag+"="+value)
	}
}
