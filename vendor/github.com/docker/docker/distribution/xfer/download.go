package xfer

import (
	"errors"
	"fmt"
	"io"
	"runtime"
	"time"

	"github.com/docker/distribution"
	"github.com/docker/docker/image"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/progress"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

const maxDownloadAttempts = 5

// LayerDownloadManager figures out which layers need to be downloaded, then
// registers and downloads those, taking into account dependencies between
// layers.
type LayerDownloadManager struct {
	layerStores  map[string]layer.Store
	tm           TransferManager
	waitDuration time.Duration
}

// SetConcurrency sets the max concurrent downloads for each pull
func (ldm *LayerDownloadManager) SetConcurrency(concurrency int) {
	ldm.tm.SetConcurrency(concurrency)
}

// NewLayerDownloadManager returns a new LayerDownloadManager.
func NewLayerDownloadManager(layerStores map[string]layer.Store, concurrencyLimit int, options ...func(*LayerDownloadManager)) *LayerDownloadManager {
	manager := LayerDownloadManager{
		layerStores:  layerStores,
		tm:           NewTransferManager(concurrencyLimit),
		waitDuration: time.Second,
	}
	for _, option := range options {
		option(&manager)
	}
	return &manager
}

type downloadTransfer struct {
	Transfer

	layerStore layer.Store
	layer      layer.Layer
	err        error
}

// result returns the layer resulting from the download, if the download
// and registration were successful.
func (d *downloadTransfer) result() (layer.Layer, error) {
	return d.layer, d.err
}

// A DownloadDescriptor references a layer that may need to be downloaded.
type DownloadDescriptor interface {
	// Key returns the key used to deduplicate downloads.
	Key() string
	// ID returns the ID for display purposes.
	ID() string
	// DiffID should return the DiffID for this layer, or an error
	// if it is unknown (for example, if it has not been downloaded
	// before).
	DiffID() (layer.DiffID, error)
	// Download is called to perform the download.
	Download(ctx context.Context, progressOutput progress.Output) (io.ReadCloser, int64, error)
	// Close is called when the download manager is finished with this
	// descriptor and will not call Download again or read from the reader
	// that Download returned.
	Close()
}

// DownloadDescriptorWithRegistered is a DownloadDescriptor that has an
// additional Registered method which gets called after a downloaded layer is
// registered. This allows the user of the download manager to know the DiffID
// of each registered layer. This method is called if a cast to
// DownloadDescriptorWithRegistered is successful.
type DownloadDescriptorWithRegistered interface {
	DownloadDescriptor
	Registered(diffID layer.DiffID)
}

// Download is a blocking function which ensures the requested layers are
// present in the layer store. It uses the string returned by the Key method to
// deduplicate downloads. If a given layer is not already known to present in
// the layer store, and the key is not used by an in-progress download, the
// Download method is called to get the layer tar data. Layers are then
// registered in the appropriate order.  The caller must call the returned
// release function once it is done with the returned RootFS object.
func (ldm *LayerDownloadManager) Download(ctx context.Context, initialRootFS image.RootFS, platform layer.Platform, layers []DownloadDescriptor, progressOutput progress.Output) (image.RootFS, func(), error) {
	var (
		topLayer       layer.Layer
		topDownload    *downloadTransfer
		watcher        *Watcher
		missingLayer   bool
		transferKey    = ""
		downloadsByKey = make(map[string]*downloadTransfer)
	)

	// Assume that the platform is the host OS if blank
	if platform == "" {
		platform = layer.Platform(runtime.GOOS)
	}

	rootFS := initialRootFS
	for _, descriptor := range layers {
		key := descriptor.Key()
		transferKey += key

		if !missingLayer {
			missingLayer = true
			diffID, err := descriptor.DiffID()
			if err == nil {
				getRootFS := rootFS
				getRootFS.Append(diffID)
				l, err := ldm.layerStores[string(platform)].Get(getRootFS.ChainID())
				if err == nil {
					// Layer already exists.
					logrus.Debugf("Layer already exists: %s", descriptor.ID())
					progress.Update(progressOutput, descriptor.ID(), "Already exists")
					if topLayer != nil {
						layer.ReleaseAndLog(ldm.layerStores[string(platform)], topLayer)
					}
					topLayer = l
					missingLayer = false
					rootFS.Append(diffID)
					// Register this repository as a source of this layer.
					withRegistered, hasRegistered := descriptor.(DownloadDescriptorWithRegistered)
					if hasRegistered {
						withRegistered.Registered(diffID)
					}
					continue
				}
			}
		}

		// Does this layer have the same data as a previous layer in
		// the stack? If so, avoid downloading it more than once.
		var topDownloadUncasted Transfer
		if existingDownload, ok := downloadsByKey[key]; ok {
			xferFunc := ldm.makeDownloadFuncFromDownload(descriptor, existingDownload, topDownload, platform)
			defer topDownload.Transfer.Release(watcher)
			topDownloadUncasted, watcher = ldm.tm.Transfer(transferKey, xferFunc, progressOutput)
			topDownload = topDownloadUncasted.(*downloadTransfer)
			continue
		}

		// Layer is not known to exist - download and register it.
		progress.Update(progressOutput, descriptor.ID(), "Pulling fs layer")

		var xferFunc DoFunc
		if topDownload != nil {
			xferFunc = ldm.makeDownloadFunc(descriptor, "", topDownload, platform)
			defer topDownload.Transfer.Release(watcher)
		} else {
			xferFunc = ldm.makeDownloadFunc(descriptor, rootFS.ChainID(), nil, platform)
		}
		topDownloadUncasted, watcher = ldm.tm.Transfer(transferKey, xferFunc, progressOutput)
		topDownload = topDownloadUncasted.(*downloadTransfer)
		downloadsByKey[key] = topDownload
	}

	if topDownload == nil {
		return rootFS, func() {
			if topLayer != nil {
				layer.ReleaseAndLog(ldm.layerStores[string(platform)], topLayer)
			}
		}, nil
	}

	// Won't be using the list built up so far - will generate it
	// from downloaded layers instead.
	rootFS.DiffIDs = []layer.DiffID{}

	defer func() {
		if topLayer != nil {
			layer.ReleaseAndLog(ldm.layerStores[string(platform)], topLayer)
		}
	}()

	select {
	case <-ctx.Done():
		topDownload.Transfer.Release(watcher)
		return rootFS, func() {}, ctx.Err()
	case <-topDownload.Done():
		break
	}

	l, err := topDownload.result()
	if err != nil {
		topDownload.Transfer.Release(watcher)
		return rootFS, func() {}, err
	}

	// Must do this exactly len(layers) times, so we don't include the
	// base layer on Windows.
	for range layers {
		if l == nil {
			topDownload.Transfer.Release(watcher)
			return rootFS, func() {}, errors.New("internal error: too few parent layers")
		}
		rootFS.DiffIDs = append([]layer.DiffID{l.DiffID()}, rootFS.DiffIDs...)
		l = l.Parent()
	}
	return rootFS, func() { topDownload.Transfer.Release(watcher) }, err
}

// makeDownloadFunc returns a function that performs the layer download and
// registration. If parentDownload is non-nil, it waits for that download to
// complete before the registration step, and registers the downloaded data
// on top of parentDownload's resulting layer. Otherwise, it registers the
// layer on top of the ChainID given by parentLayer.
func (ldm *LayerDownloadManager) makeDownloadFunc(descriptor DownloadDescriptor, parentLayer layer.ChainID, parentDownload *downloadTransfer, platform layer.Platform) DoFunc {
	return func(progressChan chan<- progress.Progress, start <-chan struct{}, inactive chan<- struct{}) Transfer {
		d := &downloadTransfer{
			Transfer:   NewTransfer(),
			layerStore: ldm.layerStores[string(platform)],
		}

		go func() {
			defer func() {
				close(progressChan)
			}()

			progressOutput := progress.ChanOutput(progressChan)

			select {
			case <-start:
			default:
				progress.Update(progressOutput, descriptor.ID(), "Waiting")
				<-start
			}

			if parentDownload != nil {
				// Did the parent download already fail or get
				// cancelled?
				select {
				case <-parentDownload.Done():
					_, err := parentDownload.result()
					if err != nil {
						d.err = err
						return
					}
				default:
				}
			}

			var (
				downloadReader io.ReadCloser
				size           int64
				err            error
				retries        int
			)

			defer descriptor.Close()

			for {
				downloadReader, size, err = descriptor.Download(d.Transfer.Context(), progressOutput)
				if err == nil {
					break
				}

				// If an error was returned because the context
				// was cancelled, we shouldn't retry.
				select {
				case <-d.Transfer.Context().Done():
					d.err = err
					return
				default:
				}

				retries++
				if _, isDNR := err.(DoNotRetry); isDNR || retries == maxDownloadAttempts {
					logrus.Errorf("Download failed: %v", err)
					d.err = err
					return
				}

				logrus.Errorf("Download failed, retrying: %v", err)
				delay := retries * 5
				ticker := time.NewTicker(ldm.waitDuration)

			selectLoop:
				for {
					progress.Updatef(progressOutput, descriptor.ID(), "Retrying in %d second%s", delay, (map[bool]string{true: "s"})[delay != 1])
					select {
					case <-ticker.C:
						delay--
						if delay == 0 {
							ticker.Stop()
							break selectLoop
						}
					case <-d.Transfer.Context().Done():
						ticker.Stop()
						d.err = errors.New("download cancelled during retry delay")
						return
					}

				}
			}

			close(inactive)

			if parentDownload != nil {
				select {
				case <-d.Transfer.Context().Done():
					d.err = errors.New("layer registration cancelled")
					downloadReader.Close()
					return
				case <-parentDownload.Done():
				}

				l, err := parentDownload.result()
				if err != nil {
					d.err = err
					downloadReader.Close()
					return
				}
				parentLayer = l.ChainID()
			}

			reader := progress.NewProgressReader(ioutils.NewCancelReadCloser(d.Transfer.Context(), downloadReader), progressOutput, size, descriptor.ID(), "Extracting")
			defer reader.Close()

			inflatedLayerData, err := archive.DecompressStream(reader)
			if err != nil {
				d.err = fmt.Errorf("could not get decompression stream: %v", err)
				return
			}

			var src distribution.Descriptor
			if fs, ok := descriptor.(distribution.Describable); ok {
				src = fs.Descriptor()
			}
			if ds, ok := d.layerStore.(layer.DescribableStore); ok {
				d.layer, err = ds.RegisterWithDescriptor(inflatedLayerData, parentLayer, platform, src)
			} else {
				d.layer, err = d.layerStore.Register(inflatedLayerData, parentLayer, platform)
			}
			if err != nil {
				select {
				case <-d.Transfer.Context().Done():
					d.err = errors.New("layer registration cancelled")
				default:
					d.err = fmt.Errorf("failed to register layer: %v", err)
				}
				return
			}

			progress.Update(progressOutput, descriptor.ID(), "Pull complete")
			withRegistered, hasRegistered := descriptor.(DownloadDescriptorWithRegistered)
			if hasRegistered {
				withRegistered.Registered(d.layer.DiffID())
			}

			// Doesn't actually need to be its own goroutine, but
			// done like this so we can defer close(c).
			go func() {
				<-d.Transfer.Released()
				if d.layer != nil {
					layer.ReleaseAndLog(d.layerStore, d.layer)
				}
			}()
		}()

		return d
	}
}

// makeDownloadFuncFromDownload returns a function that performs the layer
// registration when the layer data is coming from an existing download. It
// waits for sourceDownload and parentDownload to complete, and then
// reregisters the data from sourceDownload's top layer on top of
// parentDownload. This function does not log progress output because it would
// interfere with the progress reporting for sourceDownload, which has the same
// Key.
func (ldm *LayerDownloadManager) makeDownloadFuncFromDownload(descriptor DownloadDescriptor, sourceDownload *downloadTransfer, parentDownload *downloadTransfer, platform layer.Platform) DoFunc {
	return func(progressChan chan<- progress.Progress, start <-chan struct{}, inactive chan<- struct{}) Transfer {
		d := &downloadTransfer{
			Transfer:   NewTransfer(),
			layerStore: ldm.layerStores[string(platform)],
		}

		go func() {
			defer func() {
				close(progressChan)
			}()

			<-start

			close(inactive)

			select {
			case <-d.Transfer.Context().Done():
				d.err = errors.New("layer registration cancelled")
				return
			case <-parentDownload.Done():
			}

			l, err := parentDownload.result()
			if err != nil {
				d.err = err
				return
			}
			parentLayer := l.ChainID()

			// sourceDownload should have already finished if
			// parentDownload finished, but wait for it explicitly
			// to be sure.
			select {
			case <-d.Transfer.Context().Done():
				d.err = errors.New("layer registration cancelled")
				return
			case <-sourceDownload.Done():
			}

			l, err = sourceDownload.result()
			if err != nil {
				d.err = err
				return
			}

			layerReader, err := l.TarStream()
			if err != nil {
				d.err = err
				return
			}
			defer layerReader.Close()

			var src distribution.Descriptor
			if fs, ok := l.(distribution.Describable); ok {
				src = fs.Descriptor()
			}
			if ds, ok := d.layerStore.(layer.DescribableStore); ok {
				d.layer, err = ds.RegisterWithDescriptor(layerReader, parentLayer, platform, src)
			} else {
				d.layer, err = d.layerStore.Register(layerReader, parentLayer, platform)
			}
			if err != nil {
				d.err = fmt.Errorf("failed to register layer: %v", err)
				return
			}

			withRegistered, hasRegistered := descriptor.(DownloadDescriptorWithRegistered)
			if hasRegistered {
				withRegistered.Registered(d.layer.DiffID())
			}

			// Doesn't actually need to be its own goroutine, but
			// done like this so we can defer close(c).
			go func() {
				<-d.Transfer.Released()
				if d.layer != nil {
					layer.ReleaseAndLog(d.layerStore, d.layer)
				}
			}()
		}()

		return d
	}
}
