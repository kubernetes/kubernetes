package daemon

import (
	"fmt"
	"runtime"

	"github.com/docker/distribution/reference"
	"github.com/docker/docker/image"
)

// errImageDoesNotExist is error returned when no image can be found for a reference.
type errImageDoesNotExist struct {
	ref reference.Reference
}

func (e errImageDoesNotExist) Error() string {
	ref := e.ref
	if named, ok := ref.(reference.Named); ok {
		ref = reference.TagNameOnly(named)
	}
	return fmt.Sprintf("No such image: %s", reference.FamiliarString(ref))
}

func (e errImageDoesNotExist) NotFound() {}

// GetImageIDAndOS returns an image ID and operating system corresponding to the image referred to by
// refOrID.
func (daemon *Daemon) GetImageIDAndOS(refOrID string) (image.ID, string, error) {
	ref, err := reference.ParseAnyReference(refOrID)
	if err != nil {
		return "", "", validationError{err}
	}
	namedRef, ok := ref.(reference.Named)
	if !ok {
		digested, ok := ref.(reference.Digested)
		if !ok {
			return "", "", errImageDoesNotExist{ref}
		}
		id := image.IDFromDigest(digested.Digest())
		for platform := range daemon.stores {
			if _, err = daemon.stores[platform].imageStore.Get(id); err == nil {
				return id, platform, nil
			}
		}
		return "", "", errImageDoesNotExist{ref}
	}

	if digest, err := daemon.referenceStore.Get(namedRef); err == nil {
		// Search the image stores to get the operating system, defaulting to host OS.
		imageOS := runtime.GOOS
		id := image.IDFromDigest(digest)
		for os := range daemon.stores {
			if img, err := daemon.stores[os].imageStore.Get(id); err == nil {
				imageOS = img.OperatingSystem()
				break
			}
		}
		return id, imageOS, nil
	}

	// Search based on ID
	for os := range daemon.stores {
		if id, err := daemon.stores[os].imageStore.Search(refOrID); err == nil {
			return id, os, nil
		}
	}

	return "", "", errImageDoesNotExist{ref}
}

// GetImage returns an image corresponding to the image referred to by refOrID.
func (daemon *Daemon) GetImage(refOrID string) (*image.Image, error) {
	imgID, os, err := daemon.GetImageIDAndOS(refOrID)
	if err != nil {
		return nil, err
	}
	return daemon.stores[os].imageStore.Get(imgID)
}
