package daemon

import (
	"fmt"

	"github.com/docker/distribution/reference"
	"github.com/docker/docker/image"
	"github.com/docker/docker/pkg/stringid"
)

// ErrImageDoesNotExist is error returned when no image can be found for a reference.
type ErrImageDoesNotExist struct {
	ref reference.Reference
}

func (e ErrImageDoesNotExist) Error() string {
	ref := e.ref
	if named, ok := ref.(reference.Named); ok {
		ref = reference.TagNameOnly(named)
	}
	return fmt.Sprintf("No such image: %s", reference.FamiliarString(ref))
}

// GetImageIDAndPlatform returns an image ID and platform corresponding to the image referred to by
// refOrID.
func (daemon *Daemon) GetImageIDAndPlatform(refOrID string) (image.ID, string, error) {
	ref, err := reference.ParseAnyReference(refOrID)
	if err != nil {
		return "", "", err
	}
	namedRef, ok := ref.(reference.Named)
	if !ok {
		digested, ok := ref.(reference.Digested)
		if !ok {
			return "", "", ErrImageDoesNotExist{ref}
		}
		id := image.IDFromDigest(digested.Digest())
		for platform := range daemon.stores {
			if _, err = daemon.stores[platform].imageStore.Get(id); err == nil {
				return id, platform, nil
			}
		}
		return "", "", ErrImageDoesNotExist{ref}
	}

	for platform := range daemon.stores {
		if id, err := daemon.stores[platform].referenceStore.Get(namedRef); err == nil {
			return image.IDFromDigest(id), platform, nil
		}
	}

	// deprecated: repo:shortid https://github.com/docker/docker/pull/799
	if tagged, ok := namedRef.(reference.Tagged); ok {
		if tag := tagged.Tag(); stringid.IsShortID(stringid.TruncateID(tag)) {
			for platform := range daemon.stores {
				if id, err := daemon.stores[platform].imageStore.Search(tag); err == nil {
					for _, storeRef := range daemon.stores[platform].referenceStore.References(id.Digest()) {
						if storeRef.Name() == namedRef.Name() {
							return id, platform, nil
						}
					}
				}
			}
		}
	}

	// Search based on ID
	for platform := range daemon.stores {
		if id, err := daemon.stores[platform].imageStore.Search(refOrID); err == nil {
			return id, platform, nil
		}
	}

	return "", "", ErrImageDoesNotExist{ref}
}

// GetImage returns an image corresponding to the image referred to by refOrID.
func (daemon *Daemon) GetImage(refOrID string) (*image.Image, error) {
	imgID, platform, err := daemon.GetImageIDAndPlatform(refOrID)
	if err != nil {
		return nil, err
	}
	return daemon.stores[platform].imageStore.Get(imgID)
}
