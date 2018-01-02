package tarexport

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"runtime"

	"github.com/docker/distribution"
	"github.com/docker/distribution/reference"
	"github.com/docker/docker/image"
	"github.com/docker/docker/image/v1"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/chrootarchive"
	"github.com/docker/docker/pkg/progress"
	"github.com/docker/docker/pkg/streamformatter"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/pkg/symlink"
	"github.com/docker/docker/pkg/system"
	"github.com/opencontainers/go-digest"
	"github.com/sirupsen/logrus"
)

func (l *tarexporter) Load(inTar io.ReadCloser, outStream io.Writer, quiet bool) error {
	var progressOutput progress.Output
	if !quiet {
		progressOutput = streamformatter.NewJSONProgressOutput(outStream, false)
	}
	outStream = streamformatter.NewStdoutWriter(outStream)

	tmpDir, err := ioutil.TempDir("", "docker-import-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)

	if err := chrootarchive.Untar(inTar, tmpDir, nil); err != nil {
		return err
	}
	// read manifest, if no file then load in legacy mode
	manifestPath, err := safePath(tmpDir, manifestFileName)
	if err != nil {
		return err
	}
	manifestFile, err := os.Open(manifestPath)
	if err != nil {
		if os.IsNotExist(err) {
			return l.legacyLoad(tmpDir, outStream, progressOutput)
		}
		return err
	}
	defer manifestFile.Close()

	var manifest []manifestItem
	if err := json.NewDecoder(manifestFile).Decode(&manifest); err != nil {
		return err
	}

	var parentLinks []parentLink
	var imageIDsStr string
	var imageRefCount int

	for _, m := range manifest {
		configPath, err := safePath(tmpDir, m.Config)
		if err != nil {
			return err
		}
		config, err := ioutil.ReadFile(configPath)
		if err != nil {
			return err
		}
		img, err := image.NewFromJSON(config)
		if err != nil {
			return err
		}
		if err := checkCompatibleOS(img.OS); err != nil {
			return err
		}
		var rootFS image.RootFS
		rootFS = *img.RootFS
		rootFS.DiffIDs = nil

		if expected, actual := len(m.Layers), len(img.RootFS.DiffIDs); expected != actual {
			return fmt.Errorf("invalid manifest, layers length mismatch: expected %d, got %d", expected, actual)
		}

		// On Windows, validate the platform, defaulting to windows if not present.
		platform := layer.Platform(img.OS)
		if runtime.GOOS == "windows" {
			if platform == "" {
				platform = "windows"
			}
			if (platform != "windows") && (platform != "linux") {
				return fmt.Errorf("configuration for this image has an unsupported platform: %s", platform)
			}
		}

		for i, diffID := range img.RootFS.DiffIDs {
			layerPath, err := safePath(tmpDir, m.Layers[i])
			if err != nil {
				return err
			}
			r := rootFS
			r.Append(diffID)
			newLayer, err := l.ls.Get(r.ChainID())
			if err != nil {
				newLayer, err = l.loadLayer(layerPath, rootFS, diffID.String(), platform, m.LayerSources[diffID], progressOutput)
				if err != nil {
					return err
				}
			}
			defer layer.ReleaseAndLog(l.ls, newLayer)
			if expected, actual := diffID, newLayer.DiffID(); expected != actual {
				return fmt.Errorf("invalid diffID for layer %d: expected %q, got %q", i, expected, actual)
			}
			rootFS.Append(diffID)
		}

		imgID, err := l.is.Create(config)
		if err != nil {
			return err
		}
		imageIDsStr += fmt.Sprintf("Loaded image ID: %s\n", imgID)

		imageRefCount = 0
		for _, repoTag := range m.RepoTags {
			named, err := reference.ParseNormalizedNamed(repoTag)
			if err != nil {
				return err
			}
			ref, ok := named.(reference.NamedTagged)
			if !ok {
				return fmt.Errorf("invalid tag %q", repoTag)
			}
			l.setLoadedTag(ref, imgID.Digest(), outStream)
			outStream.Write([]byte(fmt.Sprintf("Loaded image: %s\n", reference.FamiliarString(ref))))
			imageRefCount++
		}

		parentLinks = append(parentLinks, parentLink{imgID, m.Parent})
		l.loggerImgEvent.LogImageEvent(imgID.String(), imgID.String(), "load")
	}

	for _, p := range validatedParentLinks(parentLinks) {
		if p.parentID != "" {
			if err := l.setParentID(p.id, p.parentID); err != nil {
				return err
			}
		}
	}

	if imageRefCount == 0 {
		outStream.Write([]byte(imageIDsStr))
	}

	return nil
}

func (l *tarexporter) setParentID(id, parentID image.ID) error {
	img, err := l.is.Get(id)
	if err != nil {
		return err
	}
	parent, err := l.is.Get(parentID)
	if err != nil {
		return err
	}
	if !checkValidParent(img, parent) {
		return fmt.Errorf("image %v is not a valid parent for %v", parent.ID(), img.ID())
	}
	return l.is.SetParent(id, parentID)
}

func (l *tarexporter) loadLayer(filename string, rootFS image.RootFS, id string, platform layer.Platform, foreignSrc distribution.Descriptor, progressOutput progress.Output) (layer.Layer, error) {
	// We use system.OpenSequential to use sequential file access on Windows, avoiding
	// depleting the standby list. On Linux, this equates to a regular os.Open.
	rawTar, err := system.OpenSequential(filename)
	if err != nil {
		logrus.Debugf("Error reading embedded tar: %v", err)
		return nil, err
	}
	defer rawTar.Close()

	var r io.Reader
	if progressOutput != nil {
		fileInfo, err := rawTar.Stat()
		if err != nil {
			logrus.Debugf("Error statting file: %v", err)
			return nil, err
		}

		r = progress.NewProgressReader(rawTar, progressOutput, fileInfo.Size(), stringid.TruncateID(id), "Loading layer")
	} else {
		r = rawTar
	}

	inflatedLayerData, err := archive.DecompressStream(r)
	if err != nil {
		return nil, err
	}
	defer inflatedLayerData.Close()

	if ds, ok := l.ls.(layer.DescribableStore); ok {
		return ds.RegisterWithDescriptor(inflatedLayerData, rootFS.ChainID(), platform, foreignSrc)
	}
	return l.ls.Register(inflatedLayerData, rootFS.ChainID(), platform)
}

func (l *tarexporter) setLoadedTag(ref reference.NamedTagged, imgID digest.Digest, outStream io.Writer) error {
	if prevID, err := l.rs.Get(ref); err == nil && prevID != imgID {
		fmt.Fprintf(outStream, "The image %s already exists, renaming the old one with ID %s to empty string\n", reference.FamiliarString(ref), string(prevID)) // todo: this message is wrong in case of multiple tags
	}

	if err := l.rs.AddTag(ref, imgID, true); err != nil {
		return err
	}
	return nil
}

func (l *tarexporter) legacyLoad(tmpDir string, outStream io.Writer, progressOutput progress.Output) error {
	if runtime.GOOS == "windows" {
		return errors.New("Windows does not support legacy loading of images")
	}

	legacyLoadedMap := make(map[string]image.ID)

	dirs, err := ioutil.ReadDir(tmpDir)
	if err != nil {
		return err
	}

	// every dir represents an image
	for _, d := range dirs {
		if d.IsDir() {
			if err := l.legacyLoadImage(d.Name(), tmpDir, legacyLoadedMap, progressOutput); err != nil {
				return err
			}
		}
	}

	// load tags from repositories file
	repositoriesPath, err := safePath(tmpDir, legacyRepositoriesFileName)
	if err != nil {
		return err
	}
	repositoriesFile, err := os.Open(repositoriesPath)
	if err != nil {
		return err
	}
	defer repositoriesFile.Close()

	repositories := make(map[string]map[string]string)
	if err := json.NewDecoder(repositoriesFile).Decode(&repositories); err != nil {
		return err
	}

	for name, tagMap := range repositories {
		for tag, oldID := range tagMap {
			imgID, ok := legacyLoadedMap[oldID]
			if !ok {
				return fmt.Errorf("invalid target ID: %v", oldID)
			}
			named, err := reference.ParseNormalizedNamed(name)
			if err != nil {
				return err
			}
			ref, err := reference.WithTag(named, tag)
			if err != nil {
				return err
			}
			l.setLoadedTag(ref, imgID.Digest(), outStream)
		}
	}

	return nil
}

func (l *tarexporter) legacyLoadImage(oldID, sourceDir string, loadedMap map[string]image.ID, progressOutput progress.Output) error {
	if _, loaded := loadedMap[oldID]; loaded {
		return nil
	}
	configPath, err := safePath(sourceDir, filepath.Join(oldID, legacyConfigFileName))
	if err != nil {
		return err
	}
	imageJSON, err := ioutil.ReadFile(configPath)
	if err != nil {
		logrus.Debugf("Error reading json: %v", err)
		return err
	}

	var img struct {
		OS     string
		Parent string
	}
	if err := json.Unmarshal(imageJSON, &img); err != nil {
		return err
	}

	if err := checkCompatibleOS(img.OS); err != nil {
		return err
	}

	var parentID image.ID
	if img.Parent != "" {
		for {
			var loaded bool
			if parentID, loaded = loadedMap[img.Parent]; !loaded {
				if err := l.legacyLoadImage(img.Parent, sourceDir, loadedMap, progressOutput); err != nil {
					return err
				}
			} else {
				break
			}
		}
	}

	// todo: try to connect with migrate code
	rootFS := image.NewRootFS()
	var history []image.History

	if parentID != "" {
		parentImg, err := l.is.Get(parentID)
		if err != nil {
			return err
		}

		rootFS = parentImg.RootFS
		history = parentImg.History
	}

	layerPath, err := safePath(sourceDir, filepath.Join(oldID, legacyLayerFileName))
	if err != nil {
		return err
	}
	newLayer, err := l.loadLayer(layerPath, *rootFS, oldID, "", distribution.Descriptor{}, progressOutput)
	if err != nil {
		return err
	}
	rootFS.Append(newLayer.DiffID())

	h, err := v1.HistoryFromConfig(imageJSON, false)
	if err != nil {
		return err
	}
	history = append(history, h)

	config, err := v1.MakeConfigFromV1Config(imageJSON, rootFS, history)
	if err != nil {
		return err
	}
	imgID, err := l.is.Create(config)
	if err != nil {
		return err
	}

	metadata, err := l.ls.Release(newLayer)
	layer.LogReleaseMetadata(metadata)
	if err != nil {
		return err
	}

	if parentID != "" {
		if err := l.is.SetParent(imgID, parentID); err != nil {
			return err
		}
	}

	loadedMap[oldID] = imgID
	return nil
}

func safePath(base, path string) (string, error) {
	return symlink.FollowSymlinkInScope(filepath.Join(base, path), base)
}

type parentLink struct {
	id, parentID image.ID
}

func validatedParentLinks(pl []parentLink) (ret []parentLink) {
mainloop:
	for i, p := range pl {
		ret = append(ret, p)
		for _, p2 := range pl {
			if p2.id == p.parentID && p2.id != p.id {
				continue mainloop
			}
		}
		ret[i].parentID = ""
	}
	return
}

func checkValidParent(img, parent *image.Image) bool {
	if len(img.History) == 0 && len(parent.History) == 0 {
		return true // having history is not mandatory
	}
	if len(img.History)-len(parent.History) != 1 {
		return false
	}
	for i, h := range parent.History {
		if !reflect.DeepEqual(h, img.History[i]) {
			return false
		}
	}
	return true
}

func checkCompatibleOS(os string) error {
	// TODO @jhowardmsft LCOW - revisit for simultaneous platforms
	platform := runtime.GOOS
	if system.LCOWSupported() {
		platform = "linux"
	}
	// always compatible if the OS matches; also match an empty OS
	if os == platform || os == "" {
		return nil
	}
	// for compatibility, only fail if the image or runtime OS is Windows
	if os == "windows" || platform == "windows" {
		return fmt.Errorf("cannot load %s image on %s", os, platform)
	}
	return nil
}
