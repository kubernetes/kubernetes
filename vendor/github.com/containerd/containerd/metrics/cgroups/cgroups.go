// +build linux

package cgroups

import (
	"github.com/containerd/cgroups"
	eventsapi "github.com/containerd/containerd/api/services/events/v1"
	"github.com/containerd/containerd/events"
	"github.com/containerd/containerd/linux"
	"github.com/containerd/containerd/log"
	"github.com/containerd/containerd/namespaces"
	"github.com/containerd/containerd/platforms"
	"github.com/containerd/containerd/plugin"
	"github.com/containerd/containerd/runtime"
	metrics "github.com/docker/go-metrics"
	"golang.org/x/net/context"
)

// Config for the cgroups monitor
type Config struct {
	NoPrometheus bool `toml:"no_prometheus"`
}

func init() {
	plugin.Register(&plugin.Registration{
		Type:   plugin.TaskMonitorPlugin,
		ID:     "cgroups",
		InitFn: New,
		Config: &Config{},
	})
}

// New returns a new cgroups monitor
func New(ic *plugin.InitContext) (interface{}, error) {
	var ns *metrics.Namespace
	config := ic.Config.(*Config)
	if !config.NoPrometheus {
		ns = metrics.NewNamespace("container", "", nil)
	}
	collector := newCollector(ns)
	oom, err := newOOMCollector(ns)
	if err != nil {
		return nil, err
	}
	if ns != nil {
		metrics.Register(ns)
	}
	ic.Meta.Platforms = append(ic.Meta.Platforms, platforms.DefaultSpec())
	return &cgroupsMonitor{
		collector: collector,
		oom:       oom,
		context:   ic.Context,
		publisher: ic.Events,
	}, nil
}

type cgroupsMonitor struct {
	collector *collector
	oom       *oomCollector
	context   context.Context
	publisher events.Publisher
}

func (m *cgroupsMonitor) Monitor(c runtime.Task) error {
	info := c.Info()
	t := c.(*linux.Task)
	cg, err := t.Cgroup()
	if err != nil {
		return err
	}
	if err := m.collector.Add(info.ID, info.Namespace, cg); err != nil {
		return err
	}
	return m.oom.Add(info.ID, info.Namespace, cg, m.trigger)
}

func (m *cgroupsMonitor) Stop(c runtime.Task) error {
	info := c.Info()
	t := c.(*linux.Task)

	cgroup, err := t.Cgroup()
	if err != nil {
		log.G(m.context).WithError(err).Warnf("unable to retrieve cgroup on stop")
	} else {
		m.collector.collect(info.ID, info.Namespace, cgroup, m.collector.storedMetrics, false, nil)
	}

	m.collector.Remove(info.ID, info.Namespace)
	return nil
}

func (m *cgroupsMonitor) trigger(id, namespace string, cg cgroups.Cgroup) {
	ctx := namespaces.WithNamespace(m.context, namespace)
	if err := m.publisher.Publish(ctx, runtime.TaskOOMEventTopic, &eventsapi.TaskOOM{
		ContainerID: id,
	}); err != nil {
		log.G(m.context).WithError(err).Error("post OOM event")
	}
}
