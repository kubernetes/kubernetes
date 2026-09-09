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

package kubeletplugin

import (
	"context"
	"errors"
	"fmt"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	drahealthv1 "k8s.io/kubelet/pkg/apis/dra-health/v1"
)

// This file defines the version-neutral device health reporting API that DRA
// driver authors implement through [DRAPlugin.WatchHealthStatus]. The versioned
// DRAResourceHealth gRPC API is deliberately kept out of the Go package API:
// the helper translates between the hand-written types below and whatever gRPC
// version a kubelet supports. This means driver authors do not have to adapt
// their source code when the gRPC API is promoted to a newer version.

// HealthStatus describes the health of a device as reported by a DRA driver.
type HealthStatus string

const (
	// HealthStatusUnknown indicates that the health of the device cannot be
	// determined. This is also the default when no status is set.
	HealthStatusUnknown HealthStatus = "Unknown"
	// HealthStatusHealthy indicates that the device is operating normally.
	HealthStatusHealthy HealthStatus = "Healthy"
	// HealthStatusUnhealthy indicates that the device has reported a problem.
	HealthStatusUnhealthy HealthStatus = "Unhealthy"
)

// DeviceHealth describes the health of a single device managed by a DRA driver.
type DeviceHealth struct {
	// PoolName is the name of the pool which contains the device.
	PoolName string
	// DeviceName is the name of the device within the pool.
	DeviceName string
	// Health is the current health status of the device. An empty value is
	// treated as [HealthStatusUnknown].
	Health HealthStatus
	// LastUpdated is when the driver last determined this status. The zero
	// value indicates that the time is unknown.
	LastUpdated time.Time
	// HealthCheckTimeout is the interval after which the kubelet should treat
	// the device's health as unknown if it has not received a fresh report.
	// If zero or negative, the kubelet uses a default. The value is truncated
	// to whole seconds on the wire.
	HealthCheckTimeout time.Duration
	// Message is an optional human-readable detail about the device's health.
	Message string
}

// DeviceHealthReport is a snapshot of the health of devices managed by a DRA
// driver. It usually covers all of the driver's devices, but may cover a
// subset: the kubelet reconciles each report with its internal cache, and a
// device absent from a report keeps its previously reported health until that
// goes stale (see [DeviceHealth.HealthCheckTimeout]), after which the kubelet
// reports it as unknown.
//
// A driver with more than one health source (for example an event-driven
// hardware monitor plus a polling prober for devices the monitor cannot see)
// should send one report per source, each covering only that source's
// devices. Then a resend only vouches for devices which were actually
// verified, and the failure of one source decays exactly its devices to
// unknown.
type DeviceHealthReport struct {
	// Devices is the list of devices covered by this report and their
	// current health.
	Devices []DeviceHealth
}

// ErrHealthNotSupported is returned by [DRAPlugin.WatchHealthStatus] to
// indicate that a driver does not support device health reporting. The helper
// then fails the kubelet's health stream with [codes.Unimplemented], which
// tells the kubelet to stop watching.
var ErrHealthNotSupported = errors.New("device health reporting is not supported by this driver")

// defaultKubeletHealthTimeout mirrors the kubelet's default lease for device
// health data: entries which are not refreshed within this window are
// reported as unknown.
const defaultKubeletHealthTimeout = 30 * time.Second

// healthStaleCheckInterval is how often the helper checks whether the driver
// has let its health reports go stale. A variable so tests can shorten it.
var healthStaleCheckInterval = 10 * time.Second

// healthServerBridge implements the versioned v1 DRAResourceHealth gRPC
// server by calling the version-neutral [DRAPlugin.WatchHealthStatus]
// implementation provided by the driver. The helper registers it (and, via a
// conversion wrapper, the older v1alpha1 server) so that the gRPC API version
// is fully hidden from driver authors.
type healthServerBridge struct {
	drahealthv1.UnsafeDRAResourceHealthServer
	plugin DRAPlugin
}

var _ drahealthv1.DRAResourceHealthServer = &healthServerBridge{}

func (h *healthServerBridge) NodeWatchResources(_ *drahealthv1.NodeWatchResourcesRequest, stream drahealthv1.DRAResourceHealth_NodeWatchResourcesServer) error {
	// A derived context lets us tell the driver to stop producing reports when
	// sending to the kubelet fails or the stream is otherwise done.
	ctx, cancel := context.WithCancel(stream.Context())
	defer cancel()

	reports := make(chan DeviceHealthReport)
	errCh := make(chan error, 1)
	go func() {
		defer close(reports)
		errCh <- h.plugin.WatchHealthStatus(ctx, reports)
	}()

	// The kubelet reports device health as unknown when it is not refreshed
	// within each device's health check timeout. Watch for drivers which let
	// their reports go stale and surface it as a (recoverable) error instead
	// of silently degrading. The helper deliberately does not re-send the
	// last report itself: resending is the driver's job, so that a wedged
	// monitor decays to unknown instead of being kept alive by the helper.
	staleCheck := time.NewTicker(healthStaleCheckInterval)
	defer staleCheck.Stop()
	lastReport := time.Now()
	staleAfter := defaultKubeletHealthTimeout
	staleReported := false

receive:
	for {
		select {
		case report, ok := <-reports:
			if !ok {
				break receive
			}
			lastReport = time.Now()
			staleAfter = minHealthCheckTimeout(report)
			staleReported = false
			if err := stream.Send(deviceHealthReportToV1(report)); err != nil {
				// Stop the driver and drain remaining reports so that the
				// producing goroutine can observe the cancellation and exit.
				cancel()
				for range reports { //nolint:revive // Intentionally draining.
				}
				<-errCh
				return err
			}
		case <-staleCheck.C:
			if staleReported || time.Since(lastReport) <= staleAfter {
				continue
			}
			staleReported = true
			err := fmt.Errorf("no device health report was sent for %s, the kubelet now reports device health as unknown: %w", time.Since(lastReport).Truncate(time.Second), ErrRecoverable)
			h.plugin.HandleError(ctx, err, "device health reports are stale, WatchHealthStatus must re-send within each device's HealthCheckTimeout")
		}
	}

	err := <-errCh
	if errors.Is(err, ErrHealthNotSupported) {
		// The health service is always advertised and served; a driver
		// declines health reporting by returning ErrHealthNotSupported.
		// Reporting it as Unimplemented tells the kubelet to stop watching.
		return status.Error(codes.Unimplemented, ErrHealthNotSupported.Error())
	}
	return err
}

// minHealthCheckTimeout returns the tightest lease the kubelet applies to the
// report: the smallest effective per-device health check timeout, where a
// device without an explicit positive timeout gets the kubelet's default.
// This can be longer than the default when all devices ask for a longer
// lease. Empty reports use the default.
func minHealthCheckTimeout(report DeviceHealthReport) time.Duration {
	if len(report.Devices) == 0 {
		return defaultKubeletHealthTimeout
	}
	var min time.Duration
	for _, d := range report.Devices {
		lease := d.HealthCheckTimeout
		if lease <= 0 {
			lease = defaultKubeletHealthTimeout
		}
		if min == 0 || lease < min {
			min = lease
		}
	}
	return min
}

func deviceHealthReportToV1(report DeviceHealthReport) *drahealthv1.NodeWatchResourcesResponse {
	devices := make([]*drahealthv1.DeviceHealth, 0, len(report.Devices))
	for _, d := range report.Devices {
		devices = append(devices, &drahealthv1.DeviceHealth{
			Device: &drahealthv1.DeviceIdentifier{
				PoolName:   d.PoolName,
				DeviceName: d.DeviceName,
			},
			Health:                    healthStatusToV1(d.Health),
			LastUpdatedTime:           lastUpdatedToUnix(d.LastUpdated),
			HealthCheckTimeoutSeconds: int64(d.HealthCheckTimeout / time.Second),
			Message:                   d.Message,
		})
	}
	return &drahealthv1.NodeWatchResourcesResponse{Devices: devices}
}

func lastUpdatedToUnix(t time.Time) int64 {
	if t.IsZero() {
		return 0
	}
	return t.Unix()
}

func healthStatusToV1(s HealthStatus) drahealthv1.HealthStatus {
	switch s {
	case HealthStatusHealthy:
		return drahealthv1.HealthStatus_HEALTHY
	case HealthStatusUnhealthy:
		return drahealthv1.HealthStatus_UNHEALTHY
	default:
		return drahealthv1.HealthStatus_UNKNOWN
	}
}
