package aci

import (
	"archive/tar"
	"fmt"

	"github.com/appc/spec/pkg/acirenderer"
	"github.com/appc/spec/schema/types"
	ptar "github.com/coreos/rkt/pkg/tar"
)

// Given an imageID, start with the matching image available in the store,
// build its dependency list and render it inside dir
func RenderACIWithImageID(imageID types.Hash, dir string, ap acirenderer.ACIRegistry) error {
	renderedACI, err := acirenderer.GetRenderedACIWithImageID(imageID, ap)
	if err != nil {
		return err
	}
	return renderImage(renderedACI, dir, ap)
}

// Given an image app name and optional labels, get the best matching image
// available in the store, build its dependency list and render it inside dir
func RenderACI(name types.ACName, labels types.Labels, dir string, ap acirenderer.ACIRegistry) error {
	renderedACI, err := acirenderer.GetRenderedACI(name, labels, ap)
	if err != nil {
		return err
	}
	return renderImage(renderedACI, dir, ap)
}

// Given an already populated dependency list, it will extract, under the provided
// directory, the rendered ACI
func RenderACIFromList(imgs acirenderer.Images, dir string, ap acirenderer.ACIProvider) error {
	renderedACI, err := acirenderer.GetRenderedACIFromList(imgs, ap)
	if err != nil {
		return err
	}
	return renderImage(renderedACI, dir, ap)
}

// Given a RenderedACI, it will extract, under the provided directory, the
// needed files from the right source ACI.
// The manifest will be extracted from the upper ACI.
// No file overwriting is done as it should usually be called
// providing an empty directory.
func renderImage(renderedACI acirenderer.RenderedACI, dir string, ap acirenderer.ACIProvider) error {
	for _, ra := range renderedACI {
		rs, err := ap.ReadStream(ra.Key)
		if err != nil {
			return err
		}
		defer rs.Close()
		// Overwrite is not needed. If a file needs to be overwritten then the renderedACI builder has a bug
		if err := ptar.ExtractTar(tar.NewReader(rs), dir, false, ra.FileMap); err != nil {
			return fmt.Errorf("error extracting ACI: %v", err)
		}
	}

	return nil
}
