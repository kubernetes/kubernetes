package controller

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"sync/atomic"
	"time"

	"github.com/fsnotify/fsnotify"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/options/encryptionconfig"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
)

// workqueueKey is the dummy key used to process change in encryption config file.
const workqueueKey = "key"

// Runner provides an interface to start encryption config hot reload controller.
type Runner interface {
	// Run runs the controller.
	Run(ctx context.Context)
}

// DynamicKMSEncryptionConfigContent which can dynamically handle changes in encryption config file.
type DynamicKMSEncryptionConfigContent struct {
	name string

	// filePath is the path of the file to read.
	filePath string

	// encryptionConfig stores last successfully read encryption config file content.
	encryptionConfig atomic.Value

	// queue for processing changes in encryption config file.
	queue workqueue.RateLimitingInterface

	// storageFactory allows update of transformers based on updated encryption config.
	storageFactory serverstorage.StorageFactory
}

// NewDynamicKMSEncryptionConfiguration returns controller that dynamically reacts to changes in encryption config file.
func NewDynamicKMSEncryptionConfiguration(name string, filePath string, storageFactory serverstorage.StorageFactory) (*DynamicKMSEncryptionConfigContent, error) {
	encryptionConfig := &DynamicKMSEncryptionConfigContent{
		name:           name,
		filePath:       filePath,
		storageFactory: storageFactory,
		queue:          workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), fmt.Sprintf("%s-hot-reload", name)),
	}
	err := encryptionConfig.storeInitialConfig()
	if err != nil {
		return nil, err
	}

	return encryptionConfig, nil
}

// storeInitialConfig when api-server first starts.
func (ec *DynamicKMSEncryptionConfigContent) storeInitialConfig() error {
	encryptionConfig, err := ec.readEncryptionFileContent(ec.filePath)
	if err != nil {
		return err
	}
	ec.encryptionConfig.Store(encryptionConfig)

	return nil
}

// Run starts the controller and blocks until stopCh is closed.
func (ec *DynamicKMSEncryptionConfigContent) Run(ctx context.Context) {
	defer utilruntime.HandleCrash()
	defer ec.queue.ShutDown()

	klog.InfoS("Starting controller", "name", ec.name)
	defer klog.InfoS("Shutting down controller", "name", ec.name)

	// start worker for processing content
	go wait.Until(ec.runWorker, time.Second, ctx.Done())

	// start the loop that watches the encryption config file until stopCh is closed.
	go wait.Until(func() {
		if err := ec.watchEncryptionConfigFile(ctx.Done()); err != nil {
			klog.ErrorS(err, "Failed to watch encryption config file, will retry later")
		}
	}, time.Minute, ctx.Done())

	<-ctx.Done()
}

func (ec *DynamicKMSEncryptionConfigContent) watchEncryptionConfigFile(stopCh <-chan struct{}) error {
	// Trigger a check here to ensure the content will be checked periodically even if the following watch fails.
	ec.queue.Add(workqueueKey)

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("error creating fsnotify watcher: %v", err)
	}
	defer watcher.Close()

	if err = watcher.Add(ec.filePath); err != nil {
		return fmt.Errorf("error adding watch for file %s: %v", ec.filePath, err)
	}

	// Trigger a check in case the file is updated before the watch starts.
	ec.queue.Add(workqueueKey)

	for {
		select {
		case event := <-watcher.Events:
			if err := ec.handleWatchEvent(event, watcher); err != nil {
				return err
			}
		case err := <-watcher.Errors:
			return fmt.Errorf("received fsnotify error: %v", err)
		case <-stopCh:
			return nil
		}
	}
}

func (ec *DynamicKMSEncryptionConfigContent) handleWatchEvent(event fsnotify.Event, watcher *fsnotify.Watcher) error {
	// This should be executed after restarting the watch (if applicable) to ensure no file event will be missing.
	defer ec.queue.Add(workqueueKey)

	// return if file has not removed or renamed.
	if event.Op&(fsnotify.Remove|fsnotify.Rename) == 0 {		
		return nil
	}

	if err := watcher.Remove(ec.filePath); err != nil {
		klog.InfoS("Failed to remove file watch, it may have been deleted", "file", ec.filePath, "err", err)
	}
	if err := watcher.Add(ec.filePath); err != nil {
		return fmt.Errorf("error adding watch for file %s: %v", ec.filePath, err)
	}

	return nil
}

// runWorker to process file content
func (ec *DynamicKMSEncryptionConfigContent) runWorker() {
	for ec.processNextWorkItem() {
	}
}

// processNextWorkItem processes file content when there is a message in the queue.
func (ec *DynamicKMSEncryptionConfigContent) processNextWorkItem() bool {
	// key here is dummy item in the queue to trigger file content processing.
	key, quit := ec.queue.Get()
	if quit {
		return false
	}
	defer ec.queue.Done(key)

	err := ec.loadEncryptionConfig()
	if err == nil {
		ec.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with : %v", key, err))

	// add dummy item back to the queue to trigger file content processing.
	ec.queue.AddRateLimited(key)

	return true
}

// loadEncryptionConfig processes the next set of content from the file.
func (ec *DynamicKMSEncryptionConfigContent) loadEncryptionConfig() error {
	encryptionConfig, err := ec.readEncryptionFileContent(ec.filePath)
	if err != nil {
		return err
	}
	if len(encryptionConfig) == 0 {
		return fmt.Errorf("encryption config file is empty %q", ec.name)
	}

	// check if encryptionConfig is different from the current. Do nothing if they are the same.
	if !ec.hasEncryptionConfigChanged(encryptionConfig) {
		return nil
	}

	err = ec.processEncryptionConfig(encryptionConfig)
	if err != nil {
		return err
	}

	// update local copy of recent config content only if processing is successful.
	ec.encryptionConfig.Store(encryptionConfig)
	klog.V(2).InfoS("Loaded new kms encryption config content", "name", ec.name)

	return nil
}

func (ec *DynamicKMSEncryptionConfigContent) readEncryptionFileContent(filePath string) ([]byte, error) {
	return os.ReadFile(ec.filePath)
}

// processEncryptionConfig processes the encryption config.
func (ec *DynamicKMSEncryptionConfigContent) processEncryptionConfig(encryptionConfig []byte) error {
	transformerOverrides, err := encryptionconfig.GetTransformerOverrides(ec.filePath)
	if err != nil {
		klog.Warning("KMS encryption configuration is not valid.", err)
		return err
	}

	// update transformers.
	for groupResource, transformer := range transformerOverrides {
		ec.storageFactory.UpdateMutableTransformer(groupResource, transformer)
	}

	return nil
}

// hasEncryptionConfigChanged returns true if it has changed
func (ec *DynamicKMSEncryptionConfigContent) hasEncryptionConfigChanged(newEncryptionConfig []byte) bool {
	currentConfig := ec.encryptionConfig.Load()
	if currentConfig == nil {
		return true
	}

	if bytes.Equal(newEncryptionConfig, currentConfig.([]byte)) {
		klog.V(2).InfoS("Encryption config has not changed", "name", ec.name)
		return false
	}

	return true
}
