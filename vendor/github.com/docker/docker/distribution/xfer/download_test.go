package xfer

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/docker/distribution"
	"github.com/docker/docker/image"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/progress"
	"github.com/opencontainers/go-digest"
	"golang.org/x/net/context"
)

const maxDownloadConcurrency = 3

type mockLayer struct {
	layerData bytes.Buffer
	diffID    layer.DiffID
	chainID   layer.ChainID
	parent    layer.Layer
	platform  layer.Platform
}

func (ml *mockLayer) TarStream() (io.ReadCloser, error) {
	return ioutil.NopCloser(bytes.NewBuffer(ml.layerData.Bytes())), nil
}

func (ml *mockLayer) TarStreamFrom(layer.ChainID) (io.ReadCloser, error) {
	return nil, fmt.Errorf("not implemented")
}

func (ml *mockLayer) ChainID() layer.ChainID {
	return ml.chainID
}

func (ml *mockLayer) DiffID() layer.DiffID {
	return ml.diffID
}

func (ml *mockLayer) Parent() layer.Layer {
	return ml.parent
}

func (ml *mockLayer) Size() (size int64, err error) {
	return 0, nil
}

func (ml *mockLayer) DiffSize() (size int64, err error) {
	return 0, nil
}

func (ml *mockLayer) Platform() layer.Platform {
	return ml.platform
}

func (ml *mockLayer) Metadata() (map[string]string, error) {
	return make(map[string]string), nil
}

type mockLayerStore struct {
	layers map[layer.ChainID]*mockLayer
}

func createChainIDFromParent(parent layer.ChainID, dgsts ...layer.DiffID) layer.ChainID {
	if len(dgsts) == 0 {
		return parent
	}
	if parent == "" {
		return createChainIDFromParent(layer.ChainID(dgsts[0]), dgsts[1:]...)
	}
	// H = "H(n-1) SHA256(n)"
	dgst := digest.FromBytes([]byte(string(parent) + " " + string(dgsts[0])))
	return createChainIDFromParent(layer.ChainID(dgst), dgsts[1:]...)
}

func (ls *mockLayerStore) Map() map[layer.ChainID]layer.Layer {
	layers := map[layer.ChainID]layer.Layer{}

	for k, v := range ls.layers {
		layers[k] = v
	}

	return layers
}

func (ls *mockLayerStore) Register(reader io.Reader, parentID layer.ChainID, platform layer.Platform) (layer.Layer, error) {
	return ls.RegisterWithDescriptor(reader, parentID, distribution.Descriptor{})
}

func (ls *mockLayerStore) RegisterWithDescriptor(reader io.Reader, parentID layer.ChainID, _ distribution.Descriptor) (layer.Layer, error) {
	var (
		parent layer.Layer
		err    error
	)

	if parentID != "" {
		parent, err = ls.Get(parentID)
		if err != nil {
			return nil, err
		}
	}

	l := &mockLayer{parent: parent}
	_, err = l.layerData.ReadFrom(reader)
	if err != nil {
		return nil, err
	}
	l.diffID = layer.DiffID(digest.FromBytes(l.layerData.Bytes()))
	l.chainID = createChainIDFromParent(parentID, l.diffID)

	ls.layers[l.chainID] = l
	return l, nil
}

func (ls *mockLayerStore) Get(chainID layer.ChainID) (layer.Layer, error) {
	l, ok := ls.layers[chainID]
	if !ok {
		return nil, layer.ErrLayerDoesNotExist
	}
	return l, nil
}

func (ls *mockLayerStore) Release(l layer.Layer) ([]layer.Metadata, error) {
	return []layer.Metadata{}, nil
}
func (ls *mockLayerStore) CreateRWLayer(string, layer.ChainID, *layer.CreateRWLayerOpts) (layer.RWLayer, error) {
	return nil, errors.New("not implemented")
}

func (ls *mockLayerStore) GetRWLayer(string) (layer.RWLayer, error) {
	return nil, errors.New("not implemented")
}

func (ls *mockLayerStore) ReleaseRWLayer(layer.RWLayer) ([]layer.Metadata, error) {
	return nil, errors.New("not implemented")
}
func (ls *mockLayerStore) GetMountID(string) (string, error) {
	return "", errors.New("not implemented")
}

func (ls *mockLayerStore) Cleanup() error {
	return nil
}

func (ls *mockLayerStore) DriverStatus() [][2]string {
	return [][2]string{}
}

func (ls *mockLayerStore) DriverName() string {
	return "mock"
}

type mockDownloadDescriptor struct {
	currentDownloads *int32
	id               string
	diffID           layer.DiffID
	registeredDiffID layer.DiffID
	expectedDiffID   layer.DiffID
	simulateRetries  int
}

// Key returns the key used to deduplicate downloads.
func (d *mockDownloadDescriptor) Key() string {
	return d.id
}

// ID returns the ID for display purposes.
func (d *mockDownloadDescriptor) ID() string {
	return d.id
}

// DiffID should return the DiffID for this layer, or an error
// if it is unknown (for example, if it has not been downloaded
// before).
func (d *mockDownloadDescriptor) DiffID() (layer.DiffID, error) {
	if d.diffID != "" {
		return d.diffID, nil
	}
	return "", errors.New("no diffID available")
}

func (d *mockDownloadDescriptor) Registered(diffID layer.DiffID) {
	d.registeredDiffID = diffID
}

func (d *mockDownloadDescriptor) mockTarStream() io.ReadCloser {
	// The mock implementation returns the ID repeated 5 times as a tar
	// stream instead of actual tar data. The data is ignored except for
	// computing IDs.
	return ioutil.NopCloser(bytes.NewBuffer([]byte(d.id + d.id + d.id + d.id + d.id)))
}

// Download is called to perform the download.
func (d *mockDownloadDescriptor) Download(ctx context.Context, progressOutput progress.Output) (io.ReadCloser, int64, error) {
	if d.currentDownloads != nil {
		defer atomic.AddInt32(d.currentDownloads, -1)

		if atomic.AddInt32(d.currentDownloads, 1) > maxDownloadConcurrency {
			return nil, 0, errors.New("concurrency limit exceeded")
		}
	}

	// Sleep a bit to simulate a time-consuming download.
	for i := int64(0); i <= 10; i++ {
		select {
		case <-ctx.Done():
			return nil, 0, ctx.Err()
		case <-time.After(10 * time.Millisecond):
			progressOutput.WriteProgress(progress.Progress{ID: d.ID(), Action: "Downloading", Current: i, Total: 10})
		}
	}

	if d.simulateRetries != 0 {
		d.simulateRetries--
		return nil, 0, errors.New("simulating retry")
	}

	return d.mockTarStream(), 0, nil
}

func (d *mockDownloadDescriptor) Close() {
}

func downloadDescriptors(currentDownloads *int32) []DownloadDescriptor {
	return []DownloadDescriptor{
		&mockDownloadDescriptor{
			currentDownloads: currentDownloads,
			id:               "id1",
			expectedDiffID:   layer.DiffID("sha256:68e2c75dc5c78ea9240689c60d7599766c213ae210434c53af18470ae8c53ec1"),
		},
		&mockDownloadDescriptor{
			currentDownloads: currentDownloads,
			id:               "id2",
			expectedDiffID:   layer.DiffID("sha256:64a636223116aa837973a5d9c2bdd17d9b204e4f95ac423e20e65dfbb3655473"),
		},
		&mockDownloadDescriptor{
			currentDownloads: currentDownloads,
			id:               "id3",
			expectedDiffID:   layer.DiffID("sha256:58745a8bbd669c25213e9de578c4da5c8ee1c836b3581432c2b50e38a6753300"),
		},
		&mockDownloadDescriptor{
			currentDownloads: currentDownloads,
			id:               "id2",
			expectedDiffID:   layer.DiffID("sha256:64a636223116aa837973a5d9c2bdd17d9b204e4f95ac423e20e65dfbb3655473"),
		},
		&mockDownloadDescriptor{
			currentDownloads: currentDownloads,
			id:               "id4",
			expectedDiffID:   layer.DiffID("sha256:0dfb5b9577716cc173e95af7c10289322c29a6453a1718addc00c0c5b1330936"),
			simulateRetries:  1,
		},
		&mockDownloadDescriptor{
			currentDownloads: currentDownloads,
			id:               "id5",
			expectedDiffID:   layer.DiffID("sha256:0a5f25fa1acbc647f6112a6276735d0fa01e4ee2aa7ec33015e337350e1ea23d"),
		},
	}
}

func TestSuccessfulDownload(t *testing.T) {
	// TODO Windows: Fix this unit text
	if runtime.GOOS == "windows" {
		t.Skip("Needs fixing on Windows")
	}

	layerStore := &mockLayerStore{make(map[layer.ChainID]*mockLayer)}
	lsMap := make(map[string]layer.Store)
	lsMap[runtime.GOOS] = layerStore
	ldm := NewLayerDownloadManager(lsMap, maxDownloadConcurrency, func(m *LayerDownloadManager) { m.waitDuration = time.Millisecond })

	progressChan := make(chan progress.Progress)
	progressDone := make(chan struct{})
	receivedProgress := make(map[string]progress.Progress)

	go func() {
		for p := range progressChan {
			receivedProgress[p.ID] = p
		}
		close(progressDone)
	}()

	var currentDownloads int32
	descriptors := downloadDescriptors(&currentDownloads)

	firstDescriptor := descriptors[0].(*mockDownloadDescriptor)

	// Pre-register the first layer to simulate an already-existing layer
	l, err := layerStore.Register(firstDescriptor.mockTarStream(), "", layer.Platform(runtime.GOOS))
	if err != nil {
		t.Fatal(err)
	}
	firstDescriptor.diffID = l.DiffID()

	rootFS, releaseFunc, err := ldm.Download(context.Background(), *image.NewRootFS(), layer.Platform(runtime.GOOS), descriptors, progress.ChanOutput(progressChan))
	if err != nil {
		t.Fatalf("download error: %v", err)
	}

	releaseFunc()

	close(progressChan)
	<-progressDone

	if len(rootFS.DiffIDs) != len(descriptors) {
		t.Fatal("got wrong number of diffIDs in rootfs")
	}

	for i, d := range descriptors {
		descriptor := d.(*mockDownloadDescriptor)

		if descriptor.diffID != "" {
			if receivedProgress[d.ID()].Action != "Already exists" {
				t.Fatalf("did not get 'Already exists' message for %v", d.ID())
			}
		} else if receivedProgress[d.ID()].Action != "Pull complete" {
			t.Fatalf("did not get 'Pull complete' message for %v", d.ID())
		}

		if rootFS.DiffIDs[i] != descriptor.expectedDiffID {
			t.Fatalf("rootFS item %d has the wrong diffID (expected: %v got: %v)", i, descriptor.expectedDiffID, rootFS.DiffIDs[i])
		}

		if descriptor.diffID == "" && descriptor.registeredDiffID != rootFS.DiffIDs[i] {
			t.Fatal("diffID mismatch between rootFS and Registered callback")
		}
	}
}

func TestCancelledDownload(t *testing.T) {
	layerStore := &mockLayerStore{make(map[layer.ChainID]*mockLayer)}
	lsMap := make(map[string]layer.Store)
	lsMap[runtime.GOOS] = layerStore
	ldm := NewLayerDownloadManager(lsMap, maxDownloadConcurrency, func(m *LayerDownloadManager) { m.waitDuration = time.Millisecond })

	progressChan := make(chan progress.Progress)
	progressDone := make(chan struct{})

	go func() {
		for range progressChan {
		}
		close(progressDone)
	}()

	ctx, cancel := context.WithCancel(context.Background())

	go func() {
		<-time.After(time.Millisecond)
		cancel()
	}()

	descriptors := downloadDescriptors(nil)
	_, _, err := ldm.Download(ctx, *image.NewRootFS(), layer.Platform(runtime.GOOS), descriptors, progress.ChanOutput(progressChan))
	if err != context.Canceled {
		t.Fatal("expected download to be cancelled")
	}

	close(progressChan)
	<-progressDone
}
