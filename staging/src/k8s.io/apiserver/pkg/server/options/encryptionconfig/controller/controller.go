/*
Copyright 2022 The Kubernetes Authors.

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

package controller

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/fsnotify/fsnotify"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/server/options/encryptionconfig"
	"k8s.io/apiserver/pkg/server/options/encryptionconfig/metrics"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

// workqueueKey is the dummy key used to process change in encryption config file.
const workqueueKey = "key"

// DynamicKMSEncryptionConfigContent which can dynamically handle changes in encryption config file.
type DynamicKMSEncryptionConfigContent struct {
	name string

	// filePath is the path of the file to read.
	filePath string

	// lastLoadedEncryptionConfigHash stores last successfully read encryption config file content.
	lastLoadedEncryptionConfigHash string

	// queue for processing changes in encryption config file.
	queue workqueue.RateLimitingInterface

	// dynamicTransformers updates the transformers when encryption config file changes.
	dynamicTransformers *encryptionconfig.DynamicTransformers
}

// NewDynamicEncryptionConfiguration returns controller that dynamically reacts to changes in encryption config file.
func NewDynamicEncryptionConfiguration(
	name, filePath string,
	dynamicTransformers *encryptionconfig.DynamicTransformers,
	configContentHash string,
) *DynamicKMSEncryptionConfigContent {
	encryptionConfig := &DynamicKMSEncryptionConfigContent{
		name:                           name,
		filePath:                       filePath,
		lastLoadedEncryptionConfigHash: configContentHash,
		dynamicTransformers:            dynamicTransformers,
		queue:                          workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), name),
	}
	encryptionConfig.queue.Add(workqueueKey) // to avoid missing any file changes that occur in between the initial load and Run

	return encryptionConfig
}

// Run starts the controller and blocks until stopCh is closed.
func (d *DynamicKMSEncryptionConfigContent) Run(ctx context.Context) {
	defer utilruntime.HandleCrash()
	defer d.queue.ShutDown()

	klog.InfoS("Starting controller", "name", d.name)
	defer klog.InfoS("Shutting down controller", "name", d.name)

	// start worker for processing content
	go wait.UntilWithContext(ctx, d.runWorker, time.Second)

	// start the loop that watches the encryption config file until stopCh is closed.
	go wait.UntilWithContext(ctx, func(ctx context.Context) {
		if err := d.watchEncryptionConfigFile(ctx); err != nil {
			// if there is an error while setting up or handling the watches, this will ensure that we will process the config file.
			defer d.queue.Add(workqueueKey)
			klog.ErrorS(err, "Failed to watch encryption config file, will retry later")
		}
	}, time.Second)

	<-ctx.Done()
}

func (d *DynamicKMSEncryptionConfigContent) watchEncryptionConfigFile(ctx context.Context) error {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("error creating fsnotify watcher: %w", err)
	}
	defer watcher.Close()

	if err = watcher.Add(d.filePath); err != nil {
		return fmt.Errorf("error adding watch for file %s: %w", d.filePath, err)
	}

	for {
		select {
		case event := <-watcher.Events:
			if err := d.handleWatchEvent(event, watcher); err != nil {
				return err
			}
		case err := <-watcher.Errors:
			return fmt.Errorf("received fsnotify error: %w", err)
		case <-ctx.Done():
			return nil
		}
	}
}

func (d *DynamicKMSEncryptionConfigContent) handleWatchEvent(event fsnotify.Event, watcher *fsnotify.Watcher) error {
	// This should be executed after restarting the watch (if applicable) to ensure no file event will be missing.
	defer d.queue.Add(workqueueKey)

	// return if file has not been removed or renamed.
	if event.Op&(fsnotify.Remove|fsnotify.Rename) == 0 {
		return nil
	}

	if err := watcher.Remove(d.filePath); err != nil {
		klog.V(2).InfoS("Failed to remove file watch, it may have been deleted", "file", d.filePath, "err", err)
	}
	if err := watcher.Add(d.filePath); err != nil {
		return fmt.Errorf("error adding watch for file %s: %w", d.filePath, err)
	}

	return nil
}

// runWorker to process file content
func (d *DynamicKMSEncryptionConfigContent) runWorker(ctx context.Context) {
	for d.processNextWorkItem(ctx) {
	}
}

// processNextWorkItem processes file content when there is a message in the queue.
func (d *DynamicKMSEncryptionConfigContent) processNextWorkItem(serverCtx context.Context) bool {
	// key here is dummy item in the queue to trigger file content processing.
	key, quit := d.queue.Get()
	if quit {
		return false
	}
	defer d.queue.Done(key)

	var (
		updatedEffectiveConfig  bool
		err                     error
		encryptionConfiguration *encryptionconfig.EncryptionConfiguration
		configChanged           bool
	)

	// get context to close the new transformers (on error cases and on the next reload)
	// serverCtx is attached to the API server's lifecycle so we will always close transformers on shut down
	ctx, closeTransformers := context.WithCancel(serverCtx)

	defer func() {
		// TODO can work queue metrics help here?

		if !updatedEffectiveConfig {
			// avoid leaking if we're not using the newly constructed transformers (due to an error or them not being changed)
			closeTransformers()
		}

		if updatedEffectiveConfig && err == nil {
			metrics.RecordEncryptionConfigAutomaticReloadSuccess()
		}

		if err != nil {
			metrics.RecordEncryptionConfigAutomaticReloadFailure()
			utilruntime.HandleError(fmt.Errorf("error processing encryption config file %s: %v", d.filePath, err))
			// add dummy item back to the queue to trigger file content processing.
			d.queue.AddRateLimited(key)
		}
	}()

	encryptionConfiguration, configChanged, err = d.processEncryptionConfig(ctx)
	if err != nil {
		return true
	}
	if !configChanged {
		return true
	}

	if len(encryptionConfiguration.HealthChecks) != 1 {
		err = fmt.Errorf("unexpected number of healthz checks: %d. Should have only one", len(encryptionConfiguration.HealthChecks))
		return true
	}
	// get healthz checks for all new KMS plugins.
	if err = d.validateNewTransformersHealth(ctx, encryptionConfiguration.HealthChecks[0], encryptionConfiguration.KMSCloseGracePeriod); err != nil {
		return true
	}

	// update transformers.
	// when reload=true there must always be one healthz check.
	d.dynamicTransformers.Set(
		encryptionConfiguration.Transformers,
		closeTransformers,
		encryptionConfiguration.HealthChecks[0],
		encryptionConfiguration.KMSCloseGracePeriod,
	)

	// update local copy of recent config content once update is successful.
	d.lastLoadedEncryptionConfigHash = encryptionConfiguration.EncryptionFileContentHash
	klog.V(2).InfoS("Loaded new kms encryption config content", "name", d.name)

	updatedEffectiveConfig = true
	return true
}

// loadEncryptionConfig processes the next set of content from the file.
func (d *DynamicKMSEncryptionConfigContent) processEncryptionConfig(ctx context.Context) (
	encryptionConfiguration *encryptionconfig.EncryptionConfiguration,
	configChanged bool,
	err error,
) {
	// this code path will only execute if reload=true. So passing true explicitly.
	encryptionConfiguration, err = encryptionconfig.LoadEncryptionConfig(ctx, d.filePath, true)
	if err != nil {
		return nil, false, err
	}

	// check if encryptionConfig is different from the current. Do nothing if they are the same.
	if encryptionConfiguration.EncryptionFileContentHash == d.lastLoadedEncryptionConfigHash {
		klog.V(4).InfoS("Encryption config has not changed", "name", d.name)
		return nil, false, nil
	}
	return encryptionConfiguration, true, nil
}

func (d *DynamicKMSEncryptionConfigContent) validateNewTransformersHealth(
	ctx context.Context,
	kmsPluginHealthzCheck healthz.HealthChecker,
	kmsPluginCloseGracePeriod time.Duration,
) error {
	// test if new transformers are healthy
	var healthCheckError error

	if kmsPluginCloseGracePeriod < 10*time.Second {
		kmsPluginCloseGracePeriod = 10 * time.Second
	}

	// really make sure that the immediate check does not hang
	var cancel context.CancelFunc
	ctx, cancel = context.WithTimeout(ctx, kmsPluginCloseGracePeriod)
	defer cancel()

	pollErr := wait.PollImmediateWithContext(ctx, 100*time.Millisecond, kmsPluginCloseGracePeriod, func(ctx context.Context) (bool, error) {
		// create a fake http get request to health check endpoint
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("/healthz/%s", kmsPluginHealthzCheck.Name()), nil)
		if err != nil {
			return false, err
		}

		healthCheckError = kmsPluginHealthzCheck.Check(req)
		return healthCheckError == nil, nil
	})
	if pollErr != nil {
		return fmt.Errorf("health check for new transformers failed, polling error %v: %w", pollErr, healthCheckError)
	}
	klog.V(2).InfoS("Health check succeeded")
	return nil
}
