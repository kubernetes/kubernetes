// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package basic // import "go.opentelemetry.io/otel/sdk/metric/controller/basic"

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/registry"
	export "go.opentelemetry.io/otel/sdk/export/metric"
	sdk "go.opentelemetry.io/otel/sdk/metric"
	controllerTime "go.opentelemetry.io/otel/sdk/metric/controller/time"
	"go.opentelemetry.io/otel/sdk/resource"
)

// DefaultPeriod is used for:
//
// - the minimum time between calls to Collect()
// - the timeout for Export()
// - the timeout for Collect().
const DefaultPeriod = 10 * time.Second

// ErrControllerStarted indicates that a controller was started more
// than once.
var ErrControllerStarted = fmt.Errorf("controller already started")

// Controller organizes and synchronizes collection of metric data in
// both "pull" and "push" configurations.  This supports two distinct
// modes:
//
// - Push and Pull: Start() must be called to begin calling the exporter;
//   Collect() is called periodically by a background thread after starting
//   the controller.
// - Pull-Only: Start() is optional in this case, to call Collect periodically.
//   If Start() is not called, Collect() can be called manually to initiate
//   collection
//
// The controller supports mixing push and pull access to metric data
// using the export.CheckpointSet RWLock interface.  Collection will
// be blocked by a pull request in the basic controller.
type Controller struct {
	lock         sync.Mutex
	accumulator  *sdk.Accumulator
	provider     *registry.MeterProvider
	checkpointer export.Checkpointer
	exporter     export.Exporter
	wg           sync.WaitGroup
	stopCh       chan struct{}
	clock        controllerTime.Clock
	ticker       controllerTime.Ticker

	collectPeriod  time.Duration
	collectTimeout time.Duration
	pushTimeout    time.Duration

	// collectedTime is used only in configurations with no
	// exporter, when ticker != nil.
	collectedTime time.Time
}

// New constructs a Controller using the provided checkpointer and
// options (including optional exporter) to configure a metric
// export pipeline.
func New(checkpointer export.Checkpointer, opts ...Option) *Controller {
	c := &Config{
		CollectPeriod:  DefaultPeriod,
		CollectTimeout: DefaultPeriod,
		PushTimeout:    DefaultPeriod,
	}
	for _, opt := range opts {
		opt.Apply(c)
	}
	if c.Resource == nil {
		c.Resource = resource.Default()
	}

	impl := sdk.NewAccumulator(
		checkpointer,
		c.Resource,
	)
	return &Controller{
		provider:     registry.NewMeterProvider(impl),
		accumulator:  impl,
		checkpointer: checkpointer,
		exporter:     c.Exporter,
		stopCh:       nil,
		clock:        controllerTime.RealClock{},

		collectPeriod:  c.CollectPeriod,
		collectTimeout: c.CollectTimeout,
		pushTimeout:    c.PushTimeout,
	}
}

// SetClock supports setting a mock clock for testing.  This must be
// called before Start().
func (c *Controller) SetClock(clock controllerTime.Clock) {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.clock = clock
}

// MeterProvider returns a MeterProvider instance for this controller.
func (c *Controller) MeterProvider() metric.MeterProvider {
	return c.provider
}

// Start begins a ticker that periodically collects and exports
// metrics with the configured interval.  This is required for calling
// a configured Exporter (see WithExporter) and is otherwise optional
// when only pulling metric data.
//
// The passed context is passed to Collect() and subsequently to
// asynchronous instrument callbacks.  Returns an error when the
// controller was already started.
//
// Note that it is not necessary to Start a controller when only
// pulling data; use the Collect() and ForEach() methods directly in
// this case.
func (c *Controller) Start(ctx context.Context) error {
	c.lock.Lock()
	defer c.lock.Unlock()

	if c.stopCh != nil {
		return ErrControllerStarted
	}

	c.wg.Add(1)
	c.stopCh = make(chan struct{})
	c.ticker = c.clock.Ticker(c.collectPeriod)
	go c.runTicker(ctx, c.stopCh)
	return nil
}

// Stop waits for the background goroutine to return and then collects
// and exports metrics one last time before returning.  The passed
// context is passed to the final Collect() and subsequently to the
// final asynchronous instruments.
//
// Note that Stop() will not cancel an ongoing collection or export.
func (c *Controller) Stop(ctx context.Context) error {
	c.lock.Lock()
	defer c.lock.Unlock()

	if c.stopCh == nil {
		return nil
	}

	close(c.stopCh)
	c.stopCh = nil
	c.wg.Wait()
	c.ticker.Stop()
	c.ticker = nil

	return c.collect(ctx)
}

// runTicker collection on ticker events until the stop channel is closed.
func (c *Controller) runTicker(ctx context.Context, stopCh chan struct{}) {
	defer c.wg.Done()
	for {
		select {
		case <-stopCh:
			return
		case <-c.ticker.C():
			if err := c.collect(ctx); err != nil {
				otel.Handle(err)
			}
		}
	}
}

// collect computes a checkpoint and optionally exports it.
func (c *Controller) collect(ctx context.Context) error {
	if err := c.checkpoint(ctx, func() bool {
		return true
	}); err != nil {
		return err
	}
	if c.exporter == nil {
		return nil
	}
	// Note: this is not subject to collectTimeout.  This blocks the next
	// collection despite collectTimeout because it holds a lock.
	if err := c.export(ctx); err != nil {
		return err
	}
	return nil
}

// checkpoint calls the Accumulator and Checkpointer interfaces to
// compute the CheckpointSet.  This applies the configured collection
// timeout.  Note that this does not try to cancel a Collect or Export
// when Stop() is called.
func (c *Controller) checkpoint(ctx context.Context, cond func() bool) error {
	ckpt := c.checkpointer.CheckpointSet()
	ckpt.Lock()
	defer ckpt.Unlock()

	if !cond() {
		return nil
	}
	c.checkpointer.StartCollection()

	if c.collectTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.collectTimeout)
		defer cancel()
	}

	_ = c.accumulator.Collect(ctx)

	var err error
	select {
	case <-ctx.Done():
		err = ctx.Err()
	default:
		// The context wasn't done, ok.
	}

	// Finish the checkpoint whether the accumulator timed out or not.
	if cerr := c.checkpointer.FinishCollection(); cerr != nil {
		if err == nil {
			err = cerr
		} else {
			err = fmt.Errorf("%s: %w", cerr.Error(), err)
		}
	}

	return err
}

// export calls the exporter with a read lock on the CheckpointSet,
// applying the configured export timeout.
func (c *Controller) export(ctx context.Context) error {
	ckpt := c.checkpointer.CheckpointSet()
	ckpt.RLock()
	defer ckpt.RUnlock()

	if c.pushTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.pushTimeout)
		defer cancel()
	}

	return c.exporter.Export(ctx, ckpt)
}

// Foreach gives the caller read-locked access to the current
// export.CheckpointSet.
func (c *Controller) ForEach(ks export.ExportKindSelector, f func(export.Record) error) error {
	ckpt := c.checkpointer.CheckpointSet()
	ckpt.RLock()
	defer ckpt.RUnlock()

	return ckpt.ForEach(ks, f)
}

// IsRunning returns true if the controller was started via Start(),
// indicating that the current export.CheckpointSet is being kept
// up-to-date.
func (c *Controller) IsRunning() bool {
	c.lock.Lock()
	defer c.lock.Unlock()
	return c.ticker != nil
}

// Collect requests a collection.  The collection will be skipped if
// the last collection is aged less than the configured collection
// period.
func (c *Controller) Collect(ctx context.Context) error {
	if c.IsRunning() {
		// When there's a non-nil ticker, there's a goroutine
		// computing checkpoints with the collection period.
		return ErrControllerStarted
	}

	return c.checkpoint(ctx, c.shouldCollect)
}

// shouldCollect returns true if the collector should collect now,
// based on the timestamp, the last collection time, and the
// configured period.
func (c *Controller) shouldCollect() bool {
	// This is called with the CheckpointSet exclusive
	// lock held.
	if c.collectPeriod == 0 {
		return true
	}
	now := c.clock.Now()
	if now.Sub(c.collectedTime) < c.collectPeriod {
		return false
	}
	c.collectedTime = now
	return true
}
