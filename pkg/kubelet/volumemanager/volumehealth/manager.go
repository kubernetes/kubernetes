/*
Copyright The Kubernetes Authors.

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

package volumehealth

import (
	"context"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/cache"
	"k8s.io/kubernetes/pkg/volume/csi"
)

const (
	CSIVolumeHealthy  = "VolumeHealthy"
	CSIBackendHealthy = "StorageHealthy"
)

// StatusUpdater updates pod volume health in the kubelet status cache.
type StatusUpdater interface {
	SetPodVolumeHealth(logger klog.Logger, podUID types.UID, volumeName string, conditions []v1.VolumeHealthCondition) bool
}

// CSINodeUpdater updates CSINode.Status.StorageHealth.
type CSINodeUpdater interface {
	UpdateCSINodeStorageHealth(ctx context.Context, driverName string, conditions []storagev1.StorageHealthCondition) error
}

// HealthClientFactory creates CSI health clients for registered drivers.
type HealthClientFactory func(driverName string) csi.HealthClient

// DefaultWorkers is the default number of goroutines probing volume health.
const DefaultWorkers = 5

// Manager periodically probes CSI volume and storage health and writes results
// to PodStatus.VolumeStatuses[*].HealthStatus and CSINode.Status.StorageHealth.
type Manager interface {
	Run(ctx context.Context)
}

// volumeKey keeps probes for a pod volume serialized even when its paths change.
type volumeKey struct {
	podUID          types.UID
	outerVolumeName string
}

type volumeToProbe struct {
	volumeKey
	driverName   string
	volumeHandle string
	stagingPath  string
	publishPath  string
}

type manager struct {
	statusUpdater  StatusUpdater
	probeInterval  time.Duration
	workers        int
	queue          workqueue.TypedInterface[volumeKey]
	clientFactory  HealthClientFactory
	listVolumes    func() []cache.VolumeToHealthCheck
	listDrivers    func() []string
	csiNodeUpdater func() CSINodeUpdater

	volumesMu sync.RWMutex
	volumes   map[volumeKey]volumeToProbe
}

// NewManager creates a volume health manager. probeInterval should typically be
// kubelet's VolumeStatsAggPeriod. workers is the number of concurrent volume
// probes. statusUpdater may be nil (probes still run but pod status is not written).
func NewManager(
	dsw cache.DesiredStateOfWorld,
	asw cache.ActualStateOfWorld,
	statusUpdater StatusUpdater,
	probeInterval time.Duration,
	workers int,
) Manager {
	if probeInterval <= 0 {
		probeInterval = time.Minute
	}
	m := newManager(statusUpdater, probeInterval, workers)
	m.listVolumes = func() []cache.VolumeToHealthCheck {
		return cache.GetVolumesToReportHealth(dsw, asw)
	}
	return m
}

func newManager(statusUpdater StatusUpdater, probeInterval time.Duration, workers int) *manager {
	if workers <= 0 {
		workers = DefaultWorkers
	}
	return &manager{
		statusUpdater: statusUpdater,
		probeInterval: probeInterval,
		workers:       workers,
		queue: workqueue.NewTypedWithConfig(workqueue.TypedQueueConfig[volumeKey]{
			Name: "csi_volume_health",
		}),
		clientFactory: csi.NewHealthClient,
		listDrivers:   csi.ListRegisteredDrivers,
		csiNodeUpdater: func() CSINodeUpdater {
			return csi.GetNodeInfoManager()
		},
	}
}

func (m *manager) Run(ctx context.Context) {
	registerHealthMetrics()
	logger := klog.FromContext(ctx)
	logger.Info("Starting CSI volume health manager", "probeInterval", m.probeInterval, "workers", m.workers)

	var wg sync.WaitGroup
	for range m.workers {
		wg.Go(func() { wait.UntilWithContext(ctx, m.runWorker, time.Second) })
	}
	wg.Go(func() { wait.UntilWithContext(ctx, m.enqueueVolumes, m.probeInterval) })
	wg.Go(func() { wait.UntilWithContext(ctx, m.probeStorageHealth, m.probeInterval) })

	<-ctx.Done()
	logger.Info("Shutting down CSI volume health manager")
	// In-flight probes see the cancelled context; ShutDown unblocks idle workers.
	m.queue.ShutDown()
	wg.Wait()
}

func (m *manager) enqueueVolumes(ctx context.Context) {
	volumes := make(map[volumeKey]volumeToProbe)
	for _, vol := range m.listVolumes() {
		if vol.Pod == nil || vol.OuterVolumeName == "" || vol.CSIVolumeHandle == "" {
			continue
		}
		key := volumeKey{
			podUID:          vol.Pod.UID,
			outerVolumeName: vol.OuterVolumeName,
		}
		volumes[key] = volumeToProbe{
			volumeKey:    key,
			driverName:   vol.DriverName,
			volumeHandle: vol.CSIVolumeHandle,
			stagingPath:  vol.StagingPath,
			publishPath:  vol.PublishPath,
		}
	}
	// Replace the snapshot so removed volumes are forgotten and pending keys
	// always use the latest probe parameters.
	m.saveVolumeSnapshot(volumes)
	for key := range volumes {
		m.queue.Add(key)
	}
}

func (m *manager) saveVolumeSnapshot(volumes map[volumeKey]volumeToProbe) {
	m.volumesMu.Lock()
	defer m.volumesMu.Unlock()
	m.volumes = volumes
}

func (m *manager) getVolumeToProbe(key volumeKey) (volumeToProbe, bool) {
	m.volumesMu.RLock()
	defer m.volumesMu.RUnlock()

	vol, exists := m.volumes[key]
	return vol, exists
}

func (m *manager) runWorker(ctx context.Context) {
	for m.processNextVolume(ctx) {
	}
}

func (m *manager) processNextVolume(ctx context.Context) bool {
	key, shutdown := m.queue.Get()
	if shutdown {
		return false
	}
	defer m.queue.Done(key)
	vol, exists := m.getVolumeToProbe(key)
	if exists {
		m.probeVolumeHealth(ctx, vol)
	}
	return true
}

func (m *manager) probeVolumeHealth(ctx context.Context, vol volumeToProbe) {
	logger := klog.FromContext(ctx)

	client := m.clientFactory(vol.driverName)
	supported, err := client.NodeSupportsVolumeHealth(ctx)
	if err != nil {
		logger.V(4).Info("Failed to check volume health capability",
			"driver", vol.driverName, "err", err)
		return
	}
	if !supported {
		return
	}

	start := time.Now()
	conditions, err := client.NodeGetVolumeHealth(ctx, vol.volumeHandle, vol.stagingPath, vol.publishPath)
	if err != nil {
		logger.V(4).Info("NodeGetVolumeHealth failed; leaving previous conditions unchanged",
			"driver", vol.driverName, "volume", vol.outerVolumeName, "duration", time.Since(start), "err", err)
		return
	}

	if len(conditions) == 0 {
		conditions = []v1.VolumeHealthCondition{{
			Status:  v1.VolumeHealthHealthy,
			Reason:  CSIVolumeHealthy,
			Message: "Volume is healthy",
		}}
	}

	// Dedup / PATCH suppression lives in status_manager.SetPodVolumeHealth.
	if m.statusUpdater != nil {
		m.statusUpdater.SetPodVolumeHealth(logger, vol.podUID, vol.outerVolumeName, conditions)
	} else {
		logger.V(2).Info("unable to update health of", "volume", vol.outerVolumeName, "pod", vol.podUID)
	}
}

func (m *manager) probeStorageHealth(ctx context.Context) {
	logger := klog.FromContext(ctx)
	updater := m.csiNodeUpdater()
	if updater == nil {
		return
	}
	for _, driverName := range m.listDrivers() {
		if ctx.Err() != nil {
			return
		}
		client := m.clientFactory(driverName)
		supported, err := client.NodeSupportsStorageHealth(ctx)
		if err != nil {
			logger.V(4).Info("Failed to check storage health capability",
				"driver", driverName, "err", err)
			continue
		}
		if !supported {
			continue
		}

		start := time.Now()
		backendHealth, err := client.NodeGetStorageHealth(ctx, nil)
		if err != nil {
			logger.V(4).Info("NodeGetStorageHealth failed; leaving previous conditions unchanged",
				"driver", driverName, "duration", time.Since(start), "err", err)
			continue
		}

		if len(backendHealth) == 0 {
			backendHealth = []storagev1.StorageHealthCondition{{
				Status:  storagev1.StorageHealthy,
				Reason:  CSIBackendHealthy,
				Message: "Storage subsystems are healthy",
			}}
		}

		gaugeKeys := make([]storageHealthKey, 0, len(backendHealth))
		for _, c := range backendHealth {
			gaugeKeys = append(gaugeKeys, storageHealthKey{status: string(c.Status), reason: c.Reason})
		}
		setStorageHealthGauges(driverName, gaugeKeys)

		// Dedup lives in nodeinfomanager.UpdateCSINodeStorageHealth.
		if err := updater.UpdateCSINodeStorageHealth(ctx, driverName, backendHealth); err != nil {
			logger.Error(err, "Failed to update CSINode storage health", "driver", driverName)
		}
	}
}
