package layer

import (
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/opencontainers/go-digest"
	"github.com/sirupsen/logrus"
	"github.com/vbatts/tar-split/tar/asm"
	"github.com/vbatts/tar-split/tar/storage"
)

// CreateRWLayerByGraphID creates a RWLayer in the layer store using
// the provided name with the given graphID. To get the RWLayer
// after migration the layer may be retrieved by the given name.
func (ls *layerStore) CreateRWLayerByGraphID(name string, graphID string, parent ChainID) (err error) {
	ls.mountL.Lock()
	defer ls.mountL.Unlock()
	m, ok := ls.mounts[name]
	if ok {
		if m.parent.chainID != parent {
			return errors.New("name conflict, mismatched parent")
		}
		if m.mountID != graphID {
			return errors.New("mount already exists")
		}

		return nil
	}

	if !ls.driver.Exists(graphID) {
		return fmt.Errorf("graph ID does not exist: %q", graphID)
	}

	var p *roLayer
	if string(parent) != "" {
		p = ls.get(parent)
		if p == nil {
			return ErrLayerDoesNotExist
		}

		// Release parent chain if error
		defer func() {
			if err != nil {
				ls.layerL.Lock()
				ls.releaseLayer(p)
				ls.layerL.Unlock()
			}
		}()
	}

	// TODO: Ensure graphID has correct parent

	m = &mountedLayer{
		name:       name,
		parent:     p,
		mountID:    graphID,
		layerStore: ls,
		references: map[RWLayer]*referencedRWLayer{},
	}

	// Check for existing init layer
	initID := fmt.Sprintf("%s-init", graphID)
	if ls.driver.Exists(initID) {
		m.initID = initID
	}

	if err = ls.saveMount(m); err != nil {
		return err
	}

	return nil
}

func (ls *layerStore) ChecksumForGraphID(id, parent, oldTarDataPath, newTarDataPath string) (diffID DiffID, size int64, err error) {
	defer func() {
		if err != nil {
			logrus.Debugf("could not get checksum for %q with tar-split: %q", id, err)
			diffID, size, err = ls.checksumForGraphIDNoTarsplit(id, parent, newTarDataPath)
		}
	}()

	if oldTarDataPath == "" {
		err = errors.New("no tar-split file")
		return
	}

	tarDataFile, err := os.Open(oldTarDataPath)
	if err != nil {
		return
	}
	defer tarDataFile.Close()
	uncompressed, err := gzip.NewReader(tarDataFile)
	if err != nil {
		return
	}

	dgst := digest.Canonical.Digester()
	err = ls.assembleTarTo(id, uncompressed, &size, dgst.Hash())
	if err != nil {
		return
	}

	diffID = DiffID(dgst.Digest())
	err = os.RemoveAll(newTarDataPath)
	if err != nil {
		return
	}
	err = os.Link(oldTarDataPath, newTarDataPath)

	return
}

func (ls *layerStore) checksumForGraphIDNoTarsplit(id, parent, newTarDataPath string) (diffID DiffID, size int64, err error) {
	rawarchive, err := ls.driver.Diff(id, parent)
	if err != nil {
		return
	}
	defer rawarchive.Close()

	f, err := os.Create(newTarDataPath)
	if err != nil {
		return
	}
	defer f.Close()
	mfz := gzip.NewWriter(f)
	defer mfz.Close()
	metaPacker := storage.NewJSONPacker(mfz)

	packerCounter := &packSizeCounter{metaPacker, &size}

	archive, err := asm.NewInputTarStream(rawarchive, packerCounter, nil)
	if err != nil {
		return
	}
	dgst, err := digest.FromReader(archive)
	if err != nil {
		return
	}
	diffID = DiffID(dgst)
	return
}

func (ls *layerStore) RegisterByGraphID(graphID string, parent ChainID, diffID DiffID, tarDataFile string, size int64) (Layer, error) {
	// err is used to hold the error which will always trigger
	// cleanup of creates sources but may not be an error returned
	// to the caller (already exists).
	var err error
	var p *roLayer
	if string(parent) != "" {
		p = ls.get(parent)
		if p == nil {
			return nil, ErrLayerDoesNotExist
		}

		// Release parent chain if error
		defer func() {
			if err != nil {
				ls.layerL.Lock()
				ls.releaseLayer(p)
				ls.layerL.Unlock()
			}
		}()
	}

	// Create new roLayer
	layer := &roLayer{
		parent:         p,
		cacheID:        graphID,
		referenceCount: 1,
		layerStore:     ls,
		references:     map[Layer]struct{}{},
		diffID:         diffID,
		size:           size,
		chainID:        createChainIDFromParent(parent, diffID),
	}

	ls.layerL.Lock()
	defer ls.layerL.Unlock()

	if existingLayer := ls.getWithoutLock(layer.chainID); existingLayer != nil {
		// Set error for cleanup, but do not return
		err = errors.New("layer already exists")
		return existingLayer.getReference(), nil
	}

	tx, err := ls.store.StartTransaction()
	if err != nil {
		return nil, err
	}

	defer func() {
		if err != nil {
			logrus.Debugf("Cleaning up transaction after failed migration for %s: %v", graphID, err)
			if err := tx.Cancel(); err != nil {
				logrus.Errorf("Error canceling metadata transaction %q: %s", tx.String(), err)
			}
		}
	}()

	tsw, err := tx.TarSplitWriter(false)
	if err != nil {
		return nil, err
	}
	defer tsw.Close()
	tdf, err := os.Open(tarDataFile)
	if err != nil {
		return nil, err
	}
	defer tdf.Close()
	_, err = io.Copy(tsw, tdf)
	if err != nil {
		return nil, err
	}

	if err = storeLayer(tx, layer); err != nil {
		return nil, err
	}

	if err = tx.Commit(layer.chainID); err != nil {
		return nil, err
	}

	ls.layerMap[layer.chainID] = layer

	return layer.getReference(), nil
}

type unpackSizeCounter struct {
	unpacker storage.Unpacker
	size     *int64
}

func (u *unpackSizeCounter) Next() (*storage.Entry, error) {
	e, err := u.unpacker.Next()
	if err == nil && u.size != nil {
		*u.size += e.Size
	}
	return e, err
}

type packSizeCounter struct {
	packer storage.Packer
	size   *int64
}

func (p *packSizeCounter) AddEntry(e storage.Entry) (int, error) {
	n, err := p.packer.AddEntry(e)
	if err == nil && p.size != nil {
		*p.size += e.Size
	}
	return n, err
}
