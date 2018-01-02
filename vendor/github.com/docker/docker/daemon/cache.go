package daemon

import (
	"github.com/docker/docker/builder"
	"github.com/docker/docker/image/cache"
	"github.com/sirupsen/logrus"
)

// MakeImageCache creates a stateful image cache.
func (daemon *Daemon) MakeImageCache(sourceRefs []string, platform string) builder.ImageCache {
	if len(sourceRefs) == 0 {
		return cache.NewLocal(daemon.stores[platform].imageStore)
	}

	cache := cache.New(daemon.stores[platform].imageStore)

	for _, ref := range sourceRefs {
		img, err := daemon.GetImage(ref)
		if err != nil {
			logrus.Warnf("Could not look up %s for cache resolution, skipping: %+v", ref, err)
			continue
		}
		cache.Populate(img)
	}

	return cache
}
