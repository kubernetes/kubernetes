/*
Copyright 2026 The Kubernetes Authors.

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
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/cache"
	"k8s.io/kubernetes/pkg/volume/csi"
)

// StatusUpdater updates pod volume health in the kubelet status cache.
type StatusUpdater interface {
	SetPodVolumeHealth(logger klog.Logger, podUID types.UID, volumeName string, conditions []v1.VolumeHealthCondition) bool
}

// CSINodeUpdater updates CSINode.Status.StorageHealth.
type CSINodeUpdater interface {
	UpdateCSINodeStorageHealth(driverName string, conditions []storagev1.StorageHealthCondition) error
}

// HealthClientFactory creates CSI health clients for registered drivers.
type HealthClientFactory func(driverName string) (csi.HealthClient, error)

// Manager periodically probes CSI volume and storage health and writes results
// to PodStatus.VolumeHealth and CSINode.Status.StorageHealth.
type Manager interface {
	Run(ctx context.Context)
}

type manager struct {
	dsw            cache.DesiredStateOfWorld
	asw            cache.ActualStateOfWorld
	statusUpdater  StatusUpdater
	probeInterval  time.Duration
	clientFactory  HealthClientFactory
	listDrivers    func() []string
	csiNodeUpdater func() CSINodeUpdater
}

// NewManager creates a volume health manager. probeInterval should typically be
// kubelet's VolumeStatsAggPeriod. statusUpdater may be nil (probes still run but
// pod status is not written). clientFactory/listDrivers/csiNodeUpdater default to
// the CSI plugin helpers when nil.
func NewManager(
	dsw cache.DesiredStateOfWorld,
	asw cache.ActualStateOfWorld,
	statusUpdater StatusUpdater,
	probeInterval time.Duration,
) Manager {
	if probeInterval <= 0 {
		probeInterval = time.Minute
	}
	return &manager{
		dsw:           dsw,
		asw:           asw,
		statusUpdater: statusUpdater,
		probeInterval: probeInterval,
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
	logger.Info("Starting CSI volume health manager", "probeInterval", m.probeInterval)
	go wait.UntilWithContext(ctx, m.reconcile, m.probeInterval)
	<-ctx.Done()
	logger.Info("Shutting down CSI volume health manager")
}

func (m *manager) reconcile(ctx context.Context) {
	logger := klog.FromContext(ctx)
	var wg sync.WaitGroup
	wg.Go(func() { m.probeVolumeHealth(ctx, logger) })
	wg.Go(func() { m.probeStorageHealth(ctx, logger) })
	wg.Wait()
}

func (m *manager) probeVolumeHealth(ctx context.Context, logger klog.Logger) {
	volumes := cache.GetVolumesToReportHealth(m.dsw, m.asw)

	for _, vol := range volumes {
		if vol.Pod == nil || vol.OuterVolumeName == "" || vol.CSIVolumeHandle == "" {
			continue
		}

		client, err := m.clientFactory(vol.DriverName)
		if err != nil {
			logger.V(4).Info("Skipping volume health probe; CSI client unavailable",
				"driver", vol.DriverName, "volume", vol.OuterVolumeName, "err", err)
			continue
		}
		supported, err := client.NodeSupportsVolumeHealth(ctx)
		if err != nil {
			logger.V(4).Info("Failed to check volume health capability",
				"driver", vol.DriverName, "err", err)
			continue
		}
		if !supported {
			continue
		}

		conditions, err := client.NodeGetVolumeHealth(ctx, vol.CSIVolumeHandle, vol.StagingPath, vol.PublishPath)
		if err != nil {
			logger.V(4).Info("NodeGetVolumeHealth failed; leaving previous conditions unchanged",
				"driver", vol.DriverName, "volume", vol.OuterVolumeName, "err", err)
			continue
		}

		// Dedup / PATCH suppression lives in status_manager.SetPodVolumeHealth.
		if m.statusUpdater != nil {
			m.statusUpdater.SetPodVolumeHealth(logger, vol.Pod.UID, vol.OuterVolumeName, conditions)
		}
	}
}

func (m *manager) probeStorageHealth(ctx context.Context, logger klog.Logger) {
	updater := m.csiNodeUpdater()
	if updater == nil {
		return
	}
	for _, driverName := range m.listDrivers() {
		client, err := m.clientFactory(driverName)
		if err != nil {
			logger.V(4).Info("Skipping storage health probe; CSI client unavailable",
				"driver", driverName, "err", err)
			continue
		}
		supported, err := client.NodeSupportsStorageHealth(ctx)
		if err != nil {
			logger.V(4).Info("Failed to check storage health capability",
				"driver", driverName, "err", err)
			continue
		}
		if !supported {
			continue
		}

		backendHealth, err := client.NodeGetStorageHealth(ctx, nil)
		if err != nil {
			logger.V(4).Info("NodeGetStorageHealth failed; leaving previous conditions unchanged",
				"driver", driverName, "err", err)
			continue
		}

		for i := range backendHealth {
			backendHealth[i].Name = driverName
		}

		gaugeKeys := make([]storageHealthKey, 0, len(backendHealth))
		for _, c := range backendHealth {
			gaugeKeys = append(gaugeKeys, storageHealthKey{status: string(c.Status), reason: c.Reason})
		}
		setStorageHealthGauges(driverName, gaugeKeys)

		// Dedup lives in nodeinfomanager.UpdateCSINodeStorageHealth.
		if err := updater.UpdateCSINodeStorageHealth(driverName, backendHealth); err != nil {
			logger.Error(err, "Failed to update CSINode storage health", "driver", driverName)
		}
	}
}
