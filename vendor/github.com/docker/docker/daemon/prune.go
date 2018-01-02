package daemon

import (
	"fmt"
	"regexp"
	"runtime"
	"sync/atomic"
	"time"

	"github.com/docker/distribution/reference"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/filters"
	timetypes "github.com/docker/docker/api/types/time"
	"github.com/docker/docker/image"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/directory"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/docker/runconfig"
	"github.com/docker/docker/volume"
	"github.com/docker/libnetwork"
	digest "github.com/opencontainers/go-digest"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

var (
	// errPruneRunning is returned when a prune request is received while
	// one is in progress
	errPruneRunning = fmt.Errorf("a prune operation is already running")

	containersAcceptedFilters = map[string]bool{
		"label":  true,
		"label!": true,
		"until":  true,
	}
	volumesAcceptedFilters = map[string]bool{
		"label":  true,
		"label!": true,
	}
	imagesAcceptedFilters = map[string]bool{
		"dangling": true,
		"label":    true,
		"label!":   true,
		"until":    true,
	}
	networksAcceptedFilters = map[string]bool{
		"label":  true,
		"label!": true,
		"until":  true,
	}
)

// ContainersPrune removes unused containers
func (daemon *Daemon) ContainersPrune(ctx context.Context, pruneFilters filters.Args) (*types.ContainersPruneReport, error) {
	if !atomic.CompareAndSwapInt32(&daemon.pruneRunning, 0, 1) {
		return nil, errPruneRunning
	}
	defer atomic.StoreInt32(&daemon.pruneRunning, 0)

	rep := &types.ContainersPruneReport{}

	// make sure that only accepted filters have been received
	err := pruneFilters.Validate(containersAcceptedFilters)
	if err != nil {
		return nil, err
	}

	until, err := getUntilFromPruneFilters(pruneFilters)
	if err != nil {
		return nil, err
	}

	allContainers := daemon.List()
	for _, c := range allContainers {
		select {
		case <-ctx.Done():
			logrus.Debugf("ContainersPrune operation cancelled: %#v", *rep)
			return rep, nil
		default:
		}

		if !c.IsRunning() {
			if !until.IsZero() && c.Created.After(until) {
				continue
			}
			if !matchLabels(pruneFilters, c.Config.Labels) {
				continue
			}
			cSize, _ := daemon.getSize(c.ID)
			// TODO: sets RmLink to true?
			err := daemon.ContainerRm(c.ID, &types.ContainerRmConfig{})
			if err != nil {
				logrus.Warnf("failed to prune container %s: %v", c.ID, err)
				continue
			}
			if cSize > 0 {
				rep.SpaceReclaimed += uint64(cSize)
			}
			rep.ContainersDeleted = append(rep.ContainersDeleted, c.ID)
		}
	}

	return rep, nil
}

// VolumesPrune removes unused local volumes
func (daemon *Daemon) VolumesPrune(ctx context.Context, pruneFilters filters.Args) (*types.VolumesPruneReport, error) {
	if !atomic.CompareAndSwapInt32(&daemon.pruneRunning, 0, 1) {
		return nil, errPruneRunning
	}
	defer atomic.StoreInt32(&daemon.pruneRunning, 0)

	// make sure that only accepted filters have been received
	err := pruneFilters.Validate(volumesAcceptedFilters)
	if err != nil {
		return nil, err
	}

	rep := &types.VolumesPruneReport{}

	pruneVols := func(v volume.Volume) error {
		select {
		case <-ctx.Done():
			logrus.Debugf("VolumesPrune operation cancelled: %#v", *rep)
			return ctx.Err()
		default:
		}

		name := v.Name()
		refs := daemon.volumes.Refs(v)

		if len(refs) == 0 {
			detailedVolume, ok := v.(volume.DetailedVolume)
			if ok {
				if !matchLabels(pruneFilters, detailedVolume.Labels()) {
					return nil
				}
			}
			vSize, err := directory.Size(v.Path())
			if err != nil {
				logrus.Warnf("could not determine size of volume %s: %v", name, err)
			}
			err = daemon.volumes.Remove(v)
			if err != nil {
				logrus.Warnf("could not remove volume %s: %v", name, err)
				return nil
			}
			rep.SpaceReclaimed += uint64(vSize)
			rep.VolumesDeleted = append(rep.VolumesDeleted, name)
		}

		return nil
	}

	err = daemon.traverseLocalVolumes(pruneVols)
	if err == context.Canceled {
		return rep, nil
	}

	return rep, err
}

// ImagesPrune removes unused images
func (daemon *Daemon) ImagesPrune(ctx context.Context, pruneFilters filters.Args) (*types.ImagesPruneReport, error) {
	// TODO @jhowardmsft LCOW Support: This will need revisiting later.
	platform := runtime.GOOS
	if system.LCOWSupported() {
		platform = "linux"
	}

	if !atomic.CompareAndSwapInt32(&daemon.pruneRunning, 0, 1) {
		return nil, errPruneRunning
	}
	defer atomic.StoreInt32(&daemon.pruneRunning, 0)

	// make sure that only accepted filters have been received
	err := pruneFilters.Validate(imagesAcceptedFilters)
	if err != nil {
		return nil, err
	}

	rep := &types.ImagesPruneReport{}

	danglingOnly := true
	if pruneFilters.Include("dangling") {
		if pruneFilters.ExactMatch("dangling", "false") || pruneFilters.ExactMatch("dangling", "0") {
			danglingOnly = false
		} else if !pruneFilters.ExactMatch("dangling", "true") && !pruneFilters.ExactMatch("dangling", "1") {
			return nil, fmt.Errorf("Invalid filter 'dangling=%s'", pruneFilters.Get("dangling"))
		}
	}

	until, err := getUntilFromPruneFilters(pruneFilters)
	if err != nil {
		return nil, err
	}

	var allImages map[image.ID]*image.Image
	if danglingOnly {
		allImages = daemon.stores[platform].imageStore.Heads()
	} else {
		allImages = daemon.stores[platform].imageStore.Map()
	}
	allContainers := daemon.List()
	imageRefs := map[string]bool{}
	for _, c := range allContainers {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			imageRefs[c.ID] = true
		}
	}

	// Filter intermediary images and get their unique size
	allLayers := daemon.stores[platform].layerStore.Map()
	topImages := map[image.ID]*image.Image{}
	for id, img := range allImages {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			dgst := digest.Digest(id)
			if len(daemon.stores[platform].referenceStore.References(dgst)) == 0 && len(daemon.stores[platform].imageStore.Children(id)) != 0 {
				continue
			}
			if !until.IsZero() && img.Created.After(until) {
				continue
			}
			if img.Config != nil && !matchLabels(pruneFilters, img.Config.Labels) {
				continue
			}
			topImages[id] = img
		}
	}

	canceled := false
deleteImagesLoop:
	for id := range topImages {
		select {
		case <-ctx.Done():
			// we still want to calculate freed size and return the data
			canceled = true
			break deleteImagesLoop
		default:
		}

		dgst := digest.Digest(id)
		hex := dgst.Hex()
		if _, ok := imageRefs[hex]; ok {
			continue
		}

		deletedImages := []types.ImageDeleteResponseItem{}
		refs := daemon.stores[platform].referenceStore.References(dgst)
		if len(refs) > 0 {
			shouldDelete := !danglingOnly
			if !shouldDelete {
				hasTag := false
				for _, ref := range refs {
					if _, ok := ref.(reference.NamedTagged); ok {
						hasTag = true
						break
					}
				}

				// Only delete if it's untagged (i.e. repo:<none>)
				shouldDelete = !hasTag
			}

			if shouldDelete {
				for _, ref := range refs {
					imgDel, err := daemon.ImageDelete(ref.String(), false, true)
					if err != nil {
						logrus.Warnf("could not delete reference %s: %v", ref.String(), err)
						continue
					}
					deletedImages = append(deletedImages, imgDel...)
				}
			}
		} else {
			imgDel, err := daemon.ImageDelete(hex, false, true)
			if err != nil {
				logrus.Warnf("could not delete image %s: %v", hex, err)
				continue
			}
			deletedImages = append(deletedImages, imgDel...)
		}

		rep.ImagesDeleted = append(rep.ImagesDeleted, deletedImages...)
	}

	// Compute how much space was freed
	for _, d := range rep.ImagesDeleted {
		if d.Deleted != "" {
			chid := layer.ChainID(d.Deleted)
			if l, ok := allLayers[chid]; ok {
				diffSize, err := l.DiffSize()
				if err != nil {
					logrus.Warnf("failed to get layer %s size: %v", chid, err)
					continue
				}
				rep.SpaceReclaimed += uint64(diffSize)
			}
		}
	}

	if canceled {
		logrus.Debugf("ImagesPrune operation cancelled: %#v", *rep)
	}

	return rep, nil
}

// localNetworksPrune removes unused local networks
func (daemon *Daemon) localNetworksPrune(ctx context.Context, pruneFilters filters.Args) *types.NetworksPruneReport {
	rep := &types.NetworksPruneReport{}

	until, _ := getUntilFromPruneFilters(pruneFilters)

	// When the function returns true, the walk will stop.
	l := func(nw libnetwork.Network) bool {
		select {
		case <-ctx.Done():
			// context cancelled
			return true
		default:
		}
		if nw.Info().ConfigOnly() {
			return false
		}
		if !until.IsZero() && nw.Info().Created().After(until) {
			return false
		}
		if !matchLabels(pruneFilters, nw.Info().Labels()) {
			return false
		}
		nwName := nw.Name()
		if runconfig.IsPreDefinedNetwork(nwName) {
			return false
		}
		if len(nw.Endpoints()) > 0 {
			return false
		}
		if err := daemon.DeleteNetwork(nw.ID()); err != nil {
			logrus.Warnf("could not remove local network %s: %v", nwName, err)
			return false
		}
		rep.NetworksDeleted = append(rep.NetworksDeleted, nwName)
		return false
	}
	daemon.netController.WalkNetworks(l)
	return rep
}

// clusterNetworksPrune removes unused cluster networks
func (daemon *Daemon) clusterNetworksPrune(ctx context.Context, pruneFilters filters.Args) (*types.NetworksPruneReport, error) {
	rep := &types.NetworksPruneReport{}

	until, _ := getUntilFromPruneFilters(pruneFilters)

	cluster := daemon.GetCluster()

	if !cluster.IsManager() {
		return rep, nil
	}

	networks, err := cluster.GetNetworks()
	if err != nil {
		return rep, err
	}
	networkIsInUse := regexp.MustCompile(`network ([[:alnum:]]+) is in use`)
	for _, nw := range networks {
		select {
		case <-ctx.Done():
			return rep, nil
		default:
			if nw.Ingress {
				// Routing-mesh network removal has to be explicitly invoked by user
				continue
			}
			if !until.IsZero() && nw.Created.After(until) {
				continue
			}
			if !matchLabels(pruneFilters, nw.Labels) {
				continue
			}
			// https://github.com/docker/docker/issues/24186
			// `docker network inspect` unfortunately displays ONLY those containers that are local to that node.
			// So we try to remove it anyway and check the error
			err = cluster.RemoveNetwork(nw.ID)
			if err != nil {
				// we can safely ignore the "network .. is in use" error
				match := networkIsInUse.FindStringSubmatch(err.Error())
				if len(match) != 2 || match[1] != nw.ID {
					logrus.Warnf("could not remove cluster network %s: %v", nw.Name, err)
				}
				continue
			}
			rep.NetworksDeleted = append(rep.NetworksDeleted, nw.Name)
		}
	}
	return rep, nil
}

// NetworksPrune removes unused networks
func (daemon *Daemon) NetworksPrune(ctx context.Context, pruneFilters filters.Args) (*types.NetworksPruneReport, error) {
	if !atomic.CompareAndSwapInt32(&daemon.pruneRunning, 0, 1) {
		return nil, errPruneRunning
	}
	defer atomic.StoreInt32(&daemon.pruneRunning, 0)

	// make sure that only accepted filters have been received
	err := pruneFilters.Validate(networksAcceptedFilters)
	if err != nil {
		return nil, err
	}

	if _, err := getUntilFromPruneFilters(pruneFilters); err != nil {
		return nil, err
	}

	rep := &types.NetworksPruneReport{}
	if clusterRep, err := daemon.clusterNetworksPrune(ctx, pruneFilters); err == nil {
		rep.NetworksDeleted = append(rep.NetworksDeleted, clusterRep.NetworksDeleted...)
	}

	localRep := daemon.localNetworksPrune(ctx, pruneFilters)
	rep.NetworksDeleted = append(rep.NetworksDeleted, localRep.NetworksDeleted...)

	select {
	case <-ctx.Done():
		logrus.Debugf("NetworksPrune operation cancelled: %#v", *rep)
		return rep, nil
	default:
	}

	return rep, nil
}

func getUntilFromPruneFilters(pruneFilters filters.Args) (time.Time, error) {
	until := time.Time{}
	if !pruneFilters.Include("until") {
		return until, nil
	}
	untilFilters := pruneFilters.Get("until")
	if len(untilFilters) > 1 {
		return until, fmt.Errorf("more than one until filter specified")
	}
	ts, err := timetypes.GetTimestamp(untilFilters[0], time.Now())
	if err != nil {
		return until, err
	}
	seconds, nanoseconds, err := timetypes.ParseTimestamps(ts, 0)
	if err != nil {
		return until, err
	}
	until = time.Unix(seconds, nanoseconds)
	return until, nil
}

func matchLabels(pruneFilters filters.Args, labels map[string]string) bool {
	if !pruneFilters.MatchKVList("label", labels) {
		return false
	}
	// By default MatchKVList will return true if field (like 'label!') does not exist
	// So we have to add additional Include("label!") check
	if pruneFilters.Include("label!") {
		if pruneFilters.MatchKVList("label!", labels) {
			return false
		}
	}
	return true
}
