package acirenderer

import (
	"archive/tar"
	"crypto/sha512"
	"fmt"
	"hash"
	"io"
	"io/ioutil"
	"path/filepath"
	"strings"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
)

// An ACIRegistry provides all functions of an ACIProvider plus functions to
// search for an aci and get its contents
type ACIRegistry interface {
	ACIProvider
	GetImageManifest(key string) (*schema.ImageManifest, error)
	GetACI(name types.ACName, labels types.Labels) (string, error)
}

// An ACIProvider provides functions to get an ACI contents, to convert an
// ACI hash to the key under which the ACI is known to the provider and to resolve an
// ImageID to the key under which it's known to the provider.
type ACIProvider interface {
	// Read the ACI contents stream given the key. Use ResolveKey to
	// convert an ImageID to the relative provider's key.
	ReadStream(key string) (io.ReadCloser, error)
	// Converts an ImageID to the, if existent, key under which the
	// ACI is known to the provider
	ResolveKey(key string) (string, error)
	// Converts a Hash to the provider's key
	HashToKey(h hash.Hash) string
}

// An Image contains the ImageManifest, the ACIProvider's key and its Level in
// the dependency tree.
type Image struct {
	Im    *schema.ImageManifest
	Key   string
	Level uint16
}

// Images encapsulates an ordered slice of Image structs. It represents a flat
// dependency tree.
// The upper Image should be the first in the slice with a level of 0.
// For example if A is the upper image and has two deps (in order B and C). And C has one dep (D),
// the slice (reporting the app name and excluding im and Hash) should be:
// [{A, Level: 0}, {C, Level:1}, {D, Level: 2}, {B, Level: 1}]
type Images []Image

// ACIFiles represents which files to extract for every ACI
type ACIFiles struct {
	Key     string
	FileMap map[string]struct{}
}

// RenderedACI is an (ordered) slice of ACIFiles
type RenderedACI []*ACIFiles

// GetRenderedACIWithImageID, given an imageID, starts with the matching image
// available in the store, creates the dependencies list and returns the
// RenderedACI list.
func GetRenderedACIWithImageID(imageID types.Hash, ap ACIRegistry) (RenderedACI, error) {
	imgs, err := CreateDepListFromImageID(imageID, ap)
	if err != nil {
		return nil, err
	}
	return GetRenderedACIFromList(imgs, ap)
}

// GetRenderedACI, given an image app name and optional labels, starts with the
// best matching image available in the store, creates the dependencies list
// and returns the RenderedACI list.
func GetRenderedACI(name types.ACName, labels types.Labels, ap ACIRegistry) (RenderedACI, error) {
	imgs, err := CreateDepListFromNameLabels(name, labels, ap)
	if err != nil {
		return nil, err
	}
	return GetRenderedACIFromList(imgs, ap)
}

// GetRenderedACIFromList returns the RenderedACI list. All file outside rootfs
// are excluded (at the moment only "manifest").
func GetRenderedACIFromList(imgs Images, ap ACIProvider) (RenderedACI, error) {
	if len(imgs) == 0 {
		return nil, fmt.Errorf("image list empty")
	}

	allFiles := make(map[string]struct{})
	renderedACI := RenderedACI{}

	first := true
	for i, img := range imgs {
		pwlm := getUpperPWLM(imgs, i)
		ra, err := getACIFiles(img, ap, allFiles, pwlm)
		if err != nil {
			return nil, err
		}
		// Use the manifest from the upper ACI
		if first {
			ra.FileMap["manifest"] = struct{}{}
			first = false
		}
		renderedACI = append(renderedACI, ra)
	}

	return renderedACI, nil
}

// getUpperPWLM returns the pwl at the lower level for the branch where
// img[pos] lives.
func getUpperPWLM(imgs Images, pos int) map[string]struct{} {
	var pwlm map[string]struct{}
	curlevel := imgs[pos].Level
	// Start from our position and go back ignoring the other leafs.
	for i := pos; i >= 0; i-- {
		img := imgs[i]
		if img.Level < curlevel && len(img.Im.PathWhitelist) > 0 {
			pwlm = pwlToMap(img.Im.PathWhitelist)
		}
		curlevel = img.Level
	}
	return pwlm
}

// getACIFiles returns the ACIFiles struct for the given image. All files
// outside rootfs are excluded (at the moment only "manifest").
func getACIFiles(img Image, ap ACIProvider, allFiles map[string]struct{}, pwlm map[string]struct{}) (*ACIFiles, error) {
	rs, err := ap.ReadStream(img.Key)
	if err != nil {
		return nil, err
	}
	defer rs.Close()

	hash := sha512.New()
	r := io.TeeReader(rs, hash)

	thispwlm := pwlToMap(img.Im.PathWhitelist)
	ra := &ACIFiles{FileMap: make(map[string]struct{})}
	if err = Walk(tar.NewReader(r), func(hdr *tar.Header) error {
		name := hdr.Name
		cleanName := filepath.Clean(name)

		// Ignore files outside /rootfs/ (at the moment only "manifest")
		if !strings.HasPrefix(cleanName, "rootfs/") {
			return nil
		}

		// Is the file in our PathWhiteList?
		// If the file is a directory continue also if not in PathWhiteList
		if hdr.Typeflag != tar.TypeDir {
			if len(img.Im.PathWhitelist) > 0 {
				if _, ok := thispwlm[cleanName]; !ok {
					return nil
				}
			}
		}
		// Is the file in the lower level PathWhiteList of this img branch?
		if pwlm != nil {
			if _, ok := pwlm[cleanName]; !ok {
				return nil
			}
		}
		// Is the file already provided by a previous image?
		if _, ok := allFiles[cleanName]; ok {
			return nil
		}
		ra.FileMap[cleanName] = struct{}{}
		allFiles[cleanName] = struct{}{}
		return nil
	}); err != nil {
		return nil, err
	}

	// Tar does not necessarily read the complete file, so ensure we read the entirety into the hash
	if _, err := io.Copy(ioutil.Discard, r); err != nil {
		return nil, fmt.Errorf("error reading ACI: %v", err)
	}

	if g := ap.HashToKey(hash); g != img.Key {
		return nil, fmt.Errorf("image hash does not match expected (%s != %s)", g, img.Key)
	}

	ra.Key = img.Key
	return ra, nil
}

// pwlToMap converts a pathWhiteList slice to a map for faster search
// It will also prepend "rootfs/" to the provided paths and they will be
// relative to "/" so they can be easily compared with the tar.Header.Name
// If pwl length is 0, a nil map is returned
func pwlToMap(pwl []string) map[string]struct{} {
	if len(pwl) == 0 {
		return nil
	}
	m := make(map[string]struct{}, len(pwl))
	for _, name := range pwl {
		relpath := filepath.Join("rootfs", name)
		m[relpath] = struct{}{}
	}
	return m
}

func Walk(tarReader *tar.Reader, walkFunc func(hdr *tar.Header) error) error {
	for {
		hdr, err := tarReader.Next()
		if err == io.EOF {
			// end of tar archive
			break
		}
		if err != nil {
			return fmt.Errorf("Error reading tar entry: %v", err)
		}
		if err := walkFunc(hdr); err != nil {
			return err
		}
	}
	return nil
}
