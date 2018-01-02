package tarexport

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"time"

	"github.com/docker/distribution"
	"github.com/docker/distribution/reference"
	"github.com/docker/docker/image"
	"github.com/docker/docker/image/v1"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/system"
	"github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
)

type imageDescriptor struct {
	refs     []reference.NamedTagged
	layers   []string
	image    *image.Image
	layerRef layer.Layer
}

type saveSession struct {
	*tarexporter
	outDir      string
	images      map[image.ID]*imageDescriptor
	savedLayers map[string]struct{}
	diffIDPaths map[layer.DiffID]string // cache every diffID blob to avoid duplicates
}

func (l *tarexporter) Save(names []string, outStream io.Writer) error {
	images, err := l.parseNames(names)
	if err != nil {
		return err
	}

	// Release all the image top layer references
	defer l.releaseLayerReferences(images)
	return (&saveSession{tarexporter: l, images: images}).save(outStream)
}

// parseNames will parse the image names to a map which contains image.ID to *imageDescriptor.
// Each imageDescriptor holds an image top layer reference named 'layerRef'. It is taken here, should be released later.
func (l *tarexporter) parseNames(names []string) (desc map[image.ID]*imageDescriptor, rErr error) {
	imgDescr := make(map[image.ID]*imageDescriptor)
	defer func() {
		if rErr != nil {
			l.releaseLayerReferences(imgDescr)
		}
	}()

	addAssoc := func(id image.ID, ref reference.Named) error {
		if _, ok := imgDescr[id]; !ok {
			descr := &imageDescriptor{}
			if err := l.takeLayerReference(id, descr); err != nil {
				return err
			}
			imgDescr[id] = descr
		}

		if ref != nil {
			if _, ok := ref.(reference.Canonical); ok {
				return nil
			}
			tagged, ok := reference.TagNameOnly(ref).(reference.NamedTagged)
			if !ok {
				return nil
			}

			for _, t := range imgDescr[id].refs {
				if tagged.String() == t.String() {
					return nil
				}
			}
			imgDescr[id].refs = append(imgDescr[id].refs, tagged)
		}
		return nil
	}

	for _, name := range names {
		ref, err := reference.ParseAnyReference(name)
		if err != nil {
			return nil, err
		}
		namedRef, ok := ref.(reference.Named)
		if !ok {
			// Check if digest ID reference
			if digested, ok := ref.(reference.Digested); ok {
				id := image.IDFromDigest(digested.Digest())
				if err := addAssoc(id, nil); err != nil {
					return nil, err
				}
				continue
			}
			return nil, errors.Errorf("invalid reference: %v", name)
		}

		if reference.FamiliarName(namedRef) == string(digest.Canonical) {
			imgID, err := l.is.Search(name)
			if err != nil {
				return nil, err
			}
			if err := addAssoc(imgID, nil); err != nil {
				return nil, err
			}
			continue
		}
		if reference.IsNameOnly(namedRef) {
			assocs := l.rs.ReferencesByName(namedRef)
			for _, assoc := range assocs {
				if err := addAssoc(image.IDFromDigest(assoc.ID), assoc.Ref); err != nil {
					return nil, err
				}
			}
			if len(assocs) == 0 {
				imgID, err := l.is.Search(name)
				if err != nil {
					return nil, err
				}
				if err := addAssoc(imgID, nil); err != nil {
					return nil, err
				}
			}
			continue
		}
		id, err := l.rs.Get(namedRef)
		if err != nil {
			return nil, err
		}
		if err := addAssoc(image.IDFromDigest(id), namedRef); err != nil {
			return nil, err
		}

	}
	return imgDescr, nil
}

// takeLayerReference will take/Get the image top layer reference
func (l *tarexporter) takeLayerReference(id image.ID, imgDescr *imageDescriptor) error {
	img, err := l.is.Get(id)
	if err != nil {
		return err
	}
	imgDescr.image = img
	topLayerID := img.RootFS.ChainID()
	if topLayerID == "" {
		return nil
	}
	layer, err := l.ls.Get(topLayerID)
	if err != nil {
		return err
	}
	imgDescr.layerRef = layer
	return nil
}

// releaseLayerReferences will release all the image top layer references
func (l *tarexporter) releaseLayerReferences(imgDescr map[image.ID]*imageDescriptor) error {
	for _, descr := range imgDescr {
		if descr.layerRef != nil {
			l.ls.Release(descr.layerRef)
		}
	}
	return nil
}

func (s *saveSession) save(outStream io.Writer) error {
	s.savedLayers = make(map[string]struct{})
	s.diffIDPaths = make(map[layer.DiffID]string)

	// get image json
	tempDir, err := ioutil.TempDir("", "docker-export-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tempDir)

	s.outDir = tempDir
	reposLegacy := make(map[string]map[string]string)

	var manifest []manifestItem
	var parentLinks []parentLink

	for id, imageDescr := range s.images {
		foreignSrcs, err := s.saveImage(id)
		if err != nil {
			return err
		}

		var repoTags []string
		var layers []string

		for _, ref := range imageDescr.refs {
			familiarName := reference.FamiliarName(ref)
			if _, ok := reposLegacy[familiarName]; !ok {
				reposLegacy[familiarName] = make(map[string]string)
			}
			reposLegacy[familiarName][ref.Tag()] = imageDescr.layers[len(imageDescr.layers)-1]
			repoTags = append(repoTags, reference.FamiliarString(ref))
		}

		for _, l := range imageDescr.layers {
			layers = append(layers, filepath.Join(l, legacyLayerFileName))
		}

		manifest = append(manifest, manifestItem{
			Config:       id.Digest().Hex() + ".json",
			RepoTags:     repoTags,
			Layers:       layers,
			LayerSources: foreignSrcs,
		})

		parentID, _ := s.is.GetParent(id)
		parentLinks = append(parentLinks, parentLink{id, parentID})
		s.tarexporter.loggerImgEvent.LogImageEvent(id.String(), id.String(), "save")
	}

	for i, p := range validatedParentLinks(parentLinks) {
		if p.parentID != "" {
			manifest[i].Parent = p.parentID
		}
	}

	if len(reposLegacy) > 0 {
		reposFile := filepath.Join(tempDir, legacyRepositoriesFileName)
		rf, err := os.OpenFile(reposFile, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
		if err != nil {
			return err
		}

		if err := json.NewEncoder(rf).Encode(reposLegacy); err != nil {
			rf.Close()
			return err
		}

		rf.Close()

		if err := system.Chtimes(reposFile, time.Unix(0, 0), time.Unix(0, 0)); err != nil {
			return err
		}
	}

	manifestFileName := filepath.Join(tempDir, manifestFileName)
	f, err := os.OpenFile(manifestFileName, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}

	if err := json.NewEncoder(f).Encode(manifest); err != nil {
		f.Close()
		return err
	}

	f.Close()

	if err := system.Chtimes(manifestFileName, time.Unix(0, 0), time.Unix(0, 0)); err != nil {
		return err
	}

	fs, err := archive.Tar(tempDir, archive.Uncompressed)
	if err != nil {
		return err
	}
	defer fs.Close()

	_, err = io.Copy(outStream, fs)
	return err
}

func (s *saveSession) saveImage(id image.ID) (map[layer.DiffID]distribution.Descriptor, error) {
	img := s.images[id].image
	if len(img.RootFS.DiffIDs) == 0 {
		return nil, fmt.Errorf("empty export - not implemented")
	}

	var parent digest.Digest
	var layers []string
	var foreignSrcs map[layer.DiffID]distribution.Descriptor
	for i := range img.RootFS.DiffIDs {
		v1Img := image.V1Image{
			// This is for backward compatibility used for
			// pre v1.9 docker.
			Created: time.Unix(0, 0),
		}
		if i == len(img.RootFS.DiffIDs)-1 {
			v1Img = img.V1Image
		}
		rootFS := *img.RootFS
		rootFS.DiffIDs = rootFS.DiffIDs[:i+1]
		v1ID, err := v1.CreateID(v1Img, rootFS.ChainID(), parent)
		if err != nil {
			return nil, err
		}

		v1Img.ID = v1ID.Hex()
		if parent != "" {
			v1Img.Parent = parent.Hex()
		}

		src, err := s.saveLayer(rootFS.ChainID(), v1Img, img.Created)
		if err != nil {
			return nil, err
		}
		layers = append(layers, v1Img.ID)
		parent = v1ID
		if src.Digest != "" {
			if foreignSrcs == nil {
				foreignSrcs = make(map[layer.DiffID]distribution.Descriptor)
			}
			foreignSrcs[img.RootFS.DiffIDs[i]] = src
		}
	}

	configFile := filepath.Join(s.outDir, id.Digest().Hex()+".json")
	if err := ioutil.WriteFile(configFile, img.RawJSON(), 0644); err != nil {
		return nil, err
	}
	if err := system.Chtimes(configFile, img.Created, img.Created); err != nil {
		return nil, err
	}

	s.images[id].layers = layers
	return foreignSrcs, nil
}

func (s *saveSession) saveLayer(id layer.ChainID, legacyImg image.V1Image, createdTime time.Time) (distribution.Descriptor, error) {
	if _, exists := s.savedLayers[legacyImg.ID]; exists {
		return distribution.Descriptor{}, nil
	}

	outDir := filepath.Join(s.outDir, legacyImg.ID)
	if err := os.Mkdir(outDir, 0755); err != nil {
		return distribution.Descriptor{}, err
	}

	// todo: why is this version file here?
	if err := ioutil.WriteFile(filepath.Join(outDir, legacyVersionFileName), []byte("1.0"), 0644); err != nil {
		return distribution.Descriptor{}, err
	}

	imageConfig, err := json.Marshal(legacyImg)
	if err != nil {
		return distribution.Descriptor{}, err
	}

	if err := ioutil.WriteFile(filepath.Join(outDir, legacyConfigFileName), imageConfig, 0644); err != nil {
		return distribution.Descriptor{}, err
	}

	// serialize filesystem
	layerPath := filepath.Join(outDir, legacyLayerFileName)
	l, err := s.ls.Get(id)
	if err != nil {
		return distribution.Descriptor{}, err
	}
	defer layer.ReleaseAndLog(s.ls, l)

	if oldPath, exists := s.diffIDPaths[l.DiffID()]; exists {
		relPath, err := filepath.Rel(outDir, oldPath)
		if err != nil {
			return distribution.Descriptor{}, err
		}
		if err := os.Symlink(relPath, layerPath); err != nil {
			return distribution.Descriptor{}, errors.Wrap(err, "error creating symlink while saving layer")
		}
	} else {
		// Use system.CreateSequential rather than os.Create. This ensures sequential
		// file access on Windows to avoid eating into MM standby list.
		// On Linux, this equates to a regular os.Create.
		tarFile, err := system.CreateSequential(layerPath)
		if err != nil {
			return distribution.Descriptor{}, err
		}
		defer tarFile.Close()

		arch, err := l.TarStream()
		if err != nil {
			return distribution.Descriptor{}, err
		}
		defer arch.Close()

		if _, err := io.Copy(tarFile, arch); err != nil {
			return distribution.Descriptor{}, err
		}

		for _, fname := range []string{"", legacyVersionFileName, legacyConfigFileName, legacyLayerFileName} {
			// todo: maybe save layer created timestamp?
			if err := system.Chtimes(filepath.Join(outDir, fname), createdTime, createdTime); err != nil {
				return distribution.Descriptor{}, err
			}
		}

		s.diffIDPaths[l.DiffID()] = layerPath
	}
	s.savedLayers[legacyImg.ID] = struct{}{}

	var src distribution.Descriptor
	if fs, ok := l.(distribution.Describable); ok {
		src = fs.Descriptor()
	}
	return src, nil
}
