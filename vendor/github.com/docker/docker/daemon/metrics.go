package daemon

import (
	"path/filepath"
	"sync"

	"github.com/docker/docker/pkg/mount"
	"github.com/docker/docker/pkg/plugingetter"
	"github.com/docker/go-metrics"
	"github.com/pkg/errors"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/sirupsen/logrus"
)

const metricsPluginType = "MetricsCollector"

var (
	containerActions          metrics.LabeledTimer
	containerStates           metrics.LabeledGauge
	imageActions              metrics.LabeledTimer
	networkActions            metrics.LabeledTimer
	engineInfo                metrics.LabeledGauge
	engineCpus                metrics.Gauge
	engineMemory              metrics.Gauge
	healthChecksCounter       metrics.Counter
	healthChecksFailedCounter metrics.Counter

	stateCtr *stateCounter
)

func init() {
	ns := metrics.NewNamespace("engine", "daemon", nil)
	containerActions = ns.NewLabeledTimer("container_actions", "The number of seconds it takes to process each container action", "action")
	for _, a := range []string{
		"start",
		"changes",
		"commit",
		"create",
		"delete",
	} {
		containerActions.WithValues(a).Update(0)
	}

	networkActions = ns.NewLabeledTimer("network_actions", "The number of seconds it takes to process each network action", "action")
	engineInfo = ns.NewLabeledGauge("engine", "The information related to the engine and the OS it is running on", metrics.Unit("info"),
		"version",
		"commit",
		"architecture",
		"graphdriver",
		"kernel", "os",
		"os_type",
		"daemon_id", // ID is a randomly generated unique identifier (e.g. UUID4)
	)
	engineCpus = ns.NewGauge("engine_cpus", "The number of cpus that the host system of the engine has", metrics.Unit("cpus"))
	engineMemory = ns.NewGauge("engine_memory", "The number of bytes of memory that the host system of the engine has", metrics.Bytes)
	healthChecksCounter = ns.NewCounter("health_checks", "The total number of health checks")
	healthChecksFailedCounter = ns.NewCounter("health_checks_failed", "The total number of failed health checks")
	imageActions = ns.NewLabeledTimer("image_actions", "The number of seconds it takes to process each image action", "action")

	stateCtr = newStateCounter(ns.NewDesc("container_states", "The count of containers in various states", metrics.Unit("containers"), "state"))
	ns.Add(stateCtr)

	metrics.Register(ns)
}

type stateCounter struct {
	mu     sync.Mutex
	states map[string]string
	desc   *prometheus.Desc
}

func newStateCounter(desc *prometheus.Desc) *stateCounter {
	return &stateCounter{
		states: make(map[string]string),
		desc:   desc,
	}
}

func (ctr *stateCounter) get() (running int, paused int, stopped int) {
	ctr.mu.Lock()
	defer ctr.mu.Unlock()

	states := map[string]int{
		"running": 0,
		"paused":  0,
		"stopped": 0,
	}
	for _, state := range ctr.states {
		states[state]++
	}
	return states["running"], states["paused"], states["stopped"]
}

func (ctr *stateCounter) set(id, label string) {
	ctr.mu.Lock()
	ctr.states[id] = label
	ctr.mu.Unlock()
}

func (ctr *stateCounter) del(id string) {
	ctr.mu.Lock()
	delete(ctr.states, id)
	ctr.mu.Unlock()
}

func (ctr *stateCounter) Describe(ch chan<- *prometheus.Desc) {
	ch <- ctr.desc
}

func (ctr *stateCounter) Collect(ch chan<- prometheus.Metric) {
	running, paused, stopped := ctr.get()
	ch <- prometheus.MustNewConstMetric(ctr.desc, prometheus.GaugeValue, float64(running), "running")
	ch <- prometheus.MustNewConstMetric(ctr.desc, prometheus.GaugeValue, float64(paused), "paused")
	ch <- prometheus.MustNewConstMetric(ctr.desc, prometheus.GaugeValue, float64(stopped), "stopped")
}

func (d *Daemon) cleanupMetricsPlugins() {
	ls := d.PluginStore.GetAllManagedPluginsByCap(metricsPluginType)
	var wg sync.WaitGroup
	wg.Add(len(ls))

	for _, p := range ls {
		go func() {
			defer wg.Done()
			pluginStopMetricsCollection(p)
		}()
	}
	wg.Wait()

	if d.metricsPluginListener != nil {
		d.metricsPluginListener.Close()
	}
}

type metricsPlugin struct {
	plugingetter.CompatPlugin
}

func (p metricsPlugin) sock() string {
	return "metrics.sock"
}

func (p metricsPlugin) sockBase() string {
	return filepath.Join(p.BasePath(), "run", "docker")
}

func pluginStartMetricsCollection(p plugingetter.CompatPlugin) error {
	type metricsPluginResponse struct {
		Err string
	}
	var res metricsPluginResponse
	if err := p.Client().Call(metricsPluginType+".StartMetrics", nil, &res); err != nil {
		return errors.Wrap(err, "could not start metrics plugin")
	}
	if res.Err != "" {
		return errors.New(res.Err)
	}
	return nil
}

func pluginStopMetricsCollection(p plugingetter.CompatPlugin) {
	if err := p.Client().Call(metricsPluginType+".StopMetrics", nil, nil); err != nil {
		logrus.WithError(err).WithField("name", p.Name()).Error("error stopping metrics collector")
	}

	mp := metricsPlugin{p}
	sockPath := filepath.Join(mp.sockBase(), mp.sock())
	if err := mount.Unmount(sockPath); err != nil {
		if mounted, _ := mount.Mounted(sockPath); mounted {
			logrus.WithError(err).WithField("name", p.Name()).WithField("socket", sockPath).Error("error unmounting metrics socket for plugin")
		}
	}
	return
}
