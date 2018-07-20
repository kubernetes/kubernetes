package daemon

import (
	"fmt"
	"sync/atomic"

	"golang.org/x/net/context"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/directory"
	"github.com/docker/docker/volume"
	"github.com/opencontainers/go-digest"
	"github.com/sirupsen/logrus"
)

func (daemon *Daemon) getLayerRefs(platform string) map[layer.ChainID]int {
	tmpImages := daemon.stores[platform].imageStore.Map()
	layerRefs := map[layer.ChainID]int{}
	for id, img := range tmpImages {
		dgst := digest.Digest(id)
		if len(daemon.referenceStore.References(dgst)) == 0 && len(daemon.stores[platform].imageStore.Children(id)) != 0 {
			continue
		}

		rootFS := *img.RootFS
		rootFS.DiffIDs = nil
		for _, id := range img.RootFS.DiffIDs {
			rootFS.Append(id)
			chid := rootFS.ChainID()
			layerRefs[chid]++
		}
	}

	return layerRefs
}

// SystemDiskUsage returns information about the daemon data disk usage
func (daemon *Daemon) SystemDiskUsage(ctx context.Context) (*types.DiskUsage, error) {
	if !atomic.CompareAndSwapInt32(&daemon.diskUsageRunning, 0, 1) {
		return nil, fmt.Errorf("a disk usage operation is already running")
	}
	defer atomic.StoreInt32(&daemon.diskUsageRunning, 0)

	// Retrieve container list
	allContainers, err := daemon.Containers(&types.ContainerListOptions{
		Size: true,
		All:  true,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve container list: %v", err)
	}

	// Get all top images with extra attributes
	// TODO @jhowardmsft LCOW. This may need revisiting
	allImages, err := daemon.Images(filters.NewArgs(), false, true)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve image list: %v", err)
	}

	// Get all local volumes
	allVolumes := []*types.Volume{}
	getLocalVols := func(v volume.Volume) error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			if d, ok := v.(volume.DetailedVolume); ok {
				// skip local volumes with mount options since these could have external
				// mounted filesystems that will be slow to enumerate.
				if len(d.Options()) > 0 {
					return nil
				}
			}
			name := v.Name()
			refs := daemon.volumes.Refs(v)

			tv := volumeToAPIType(v)
			sz, err := directory.Size(v.Path())
			if err != nil {
				logrus.Warnf("failed to determine size of volume %v", name)
				sz = -1
			}
			tv.UsageData = &types.VolumeUsageData{Size: sz, RefCount: int64(len(refs))}
			allVolumes = append(allVolumes, tv)
		}

		return nil
	}

	err = daemon.traverseLocalVolumes(getLocalVols)
	if err != nil {
		return nil, err
	}

	// Get total layers size on disk
	var allLayersSize int64
	for platform := range daemon.stores {
		layerRefs := daemon.getLayerRefs(platform)
		allLayers := daemon.stores[platform].layerStore.Map()
		for _, l := range allLayers {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			default:
				size, err := l.DiffSize()
				if err == nil {
					if _, ok := layerRefs[l.ChainID()]; ok {
						allLayersSize += size
					} else {
						logrus.Warnf("found leaked image layer %v platform %s", l.ChainID(), platform)
					}
				} else {
					logrus.Warnf("failed to get diff size for layer %v %s", l.ChainID(), platform)
				}
			}
		}
	}

	return &types.DiskUsage{
		LayersSize: allLayersSize,
		Containers: allContainers,
		Volumes:    allVolumes,
		Images:     allImages,
	}, nil
}
