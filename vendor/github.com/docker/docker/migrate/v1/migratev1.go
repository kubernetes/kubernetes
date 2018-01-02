package v1

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"time"

	"encoding/json"

	"github.com/docker/distribution/reference"
	"github.com/docker/docker/distribution/metadata"
	"github.com/docker/docker/image"
	imagev1 "github.com/docker/docker/image/v1"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/ioutils"
	refstore "github.com/docker/docker/reference"
	"github.com/opencontainers/go-digest"
	"github.com/sirupsen/logrus"
)

type graphIDRegistrar interface {
	RegisterByGraphID(string, layer.ChainID, layer.DiffID, string, int64) (layer.Layer, error)
	Release(layer.Layer) ([]layer.Metadata, error)
}

type graphIDMounter interface {
	CreateRWLayerByGraphID(string, string, layer.ChainID) error
}

type checksumCalculator interface {
	ChecksumForGraphID(id, parent, oldTarDataPath, newTarDataPath string) (diffID layer.DiffID, size int64, err error)
}

const (
	graphDirName                 = "graph"
	tarDataFileName              = "tar-data.json.gz"
	migrationFileName            = ".migration-v1-images.json"
	migrationTagsFileName        = ".migration-v1-tags"
	migrationDiffIDFileName      = ".migration-diffid"
	migrationSizeFileName        = ".migration-size"
	migrationTarDataFileName     = ".migration-tardata"
	containersDirName            = "containers"
	configFileNameLegacy         = "config.json"
	configFileName               = "config.v2.json"
	repositoriesFilePrefixLegacy = "repositories-"
)

var (
	errUnsupported = errors.New("migration is not supported")
)

// Migrate takes an old graph directory and transforms the metadata into the
// new format.
func Migrate(root, driverName string, ls layer.Store, is image.Store, rs refstore.Store, ms metadata.Store) error {
	graphDir := filepath.Join(root, graphDirName)
	if _, err := os.Lstat(graphDir); os.IsNotExist(err) {
		return nil
	}

	mappings, err := restoreMappings(root)
	if err != nil {
		return err
	}

	if cc, ok := ls.(checksumCalculator); ok {
		CalculateLayerChecksums(root, cc, mappings)
	}

	if registrar, ok := ls.(graphIDRegistrar); !ok {
		return errUnsupported
	} else if err := migrateImages(root, registrar, is, ms, mappings); err != nil {
		return err
	}

	err = saveMappings(root, mappings)
	if err != nil {
		return err
	}

	if mounter, ok := ls.(graphIDMounter); !ok {
		return errUnsupported
	} else if err := migrateContainers(root, mounter, is, mappings); err != nil {
		return err
	}

	if err := migrateRefs(root, driverName, rs, mappings); err != nil {
		return err
	}

	return nil
}

// CalculateLayerChecksums walks an old graph directory and calculates checksums
// for each layer. These checksums are later used for migration.
func CalculateLayerChecksums(root string, ls checksumCalculator, mappings map[string]image.ID) {
	graphDir := filepath.Join(root, graphDirName)
	// spawn some extra workers also for maximum performance because the process is bounded by both cpu and io
	workers := runtime.NumCPU() * 3
	workQueue := make(chan string, workers)

	wg := sync.WaitGroup{}

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			for id := range workQueue {
				start := time.Now()
				if err := calculateLayerChecksum(graphDir, id, ls); err != nil {
					logrus.Errorf("could not calculate checksum for %q, %q", id, err)
				}
				elapsed := time.Since(start)
				logrus.Debugf("layer %s took %.2f seconds", id, elapsed.Seconds())
			}
			wg.Done()
		}()
	}

	dir, err := ioutil.ReadDir(graphDir)
	if err != nil {
		logrus.Errorf("could not read directory %q", graphDir)
		return
	}
	for _, v := range dir {
		v1ID := v.Name()
		if err := imagev1.ValidateID(v1ID); err != nil {
			continue
		}
		if _, ok := mappings[v1ID]; ok { // support old migrations without helper files
			continue
		}
		workQueue <- v1ID
	}
	close(workQueue)
	wg.Wait()
}

func calculateLayerChecksum(graphDir, id string, ls checksumCalculator) error {
	diffIDFile := filepath.Join(graphDir, id, migrationDiffIDFileName)
	if _, err := os.Lstat(diffIDFile); err == nil {
		return nil
	} else if !os.IsNotExist(err) {
		return err
	}

	parent, err := getParent(filepath.Join(graphDir, id))
	if err != nil {
		return err
	}

	diffID, size, err := ls.ChecksumForGraphID(id, parent, filepath.Join(graphDir, id, tarDataFileName), filepath.Join(graphDir, id, migrationTarDataFileName))
	if err != nil {
		return err
	}

	if err := ioutil.WriteFile(filepath.Join(graphDir, id, migrationSizeFileName), []byte(strconv.Itoa(int(size))), 0600); err != nil {
		return err
	}

	if err := ioutils.AtomicWriteFile(filepath.Join(graphDir, id, migrationDiffIDFileName), []byte(diffID), 0600); err != nil {
		return err
	}

	logrus.Infof("calculated checksum for layer %s: %s", id, diffID)
	return nil
}

func restoreMappings(root string) (map[string]image.ID, error) {
	mappings := make(map[string]image.ID)

	mfile := filepath.Join(root, migrationFileName)
	f, err := os.Open(mfile)
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	} else if err == nil {
		err := json.NewDecoder(f).Decode(&mappings)
		if err != nil {
			f.Close()
			return nil, err
		}
		f.Close()
	}

	return mappings, nil
}

func saveMappings(root string, mappings map[string]image.ID) error {
	mfile := filepath.Join(root, migrationFileName)
	f, err := os.OpenFile(mfile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0600)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(mappings)
}

func migrateImages(root string, ls graphIDRegistrar, is image.Store, ms metadata.Store, mappings map[string]image.ID) error {
	graphDir := filepath.Join(root, graphDirName)

	dir, err := ioutil.ReadDir(graphDir)
	if err != nil {
		return err
	}
	for _, v := range dir {
		v1ID := v.Name()
		if err := imagev1.ValidateID(v1ID); err != nil {
			continue
		}
		if _, exists := mappings[v1ID]; exists {
			continue
		}
		if err := migrateImage(v1ID, root, ls, is, ms, mappings); err != nil {
			continue
		}
	}

	return nil
}

func migrateContainers(root string, ls graphIDMounter, is image.Store, imageMappings map[string]image.ID) error {
	containersDir := filepath.Join(root, containersDirName)
	dir, err := ioutil.ReadDir(containersDir)
	if err != nil {
		return err
	}
	for _, v := range dir {
		id := v.Name()

		if _, err := os.Stat(filepath.Join(containersDir, id, configFileName)); err == nil {
			continue
		}

		containerJSON, err := ioutil.ReadFile(filepath.Join(containersDir, id, configFileNameLegacy))
		if err != nil {
			logrus.Errorf("migrate container error: %v", err)
			continue
		}

		var c map[string]*json.RawMessage
		if err := json.Unmarshal(containerJSON, &c); err != nil {
			logrus.Errorf("migrate container error: %v", err)
			continue
		}

		imageStrJSON, ok := c["Image"]
		if !ok {
			return fmt.Errorf("invalid container configuration for %v", id)
		}

		var image string
		if err := json.Unmarshal([]byte(*imageStrJSON), &image); err != nil {
			logrus.Errorf("migrate container error: %v", err)
			continue
		}

		imageID, ok := imageMappings[image]
		if !ok {
			logrus.Errorf("image not migrated %v", imageID) // non-fatal error
			continue
		}

		c["Image"] = rawJSON(imageID)

		containerJSON, err = json.Marshal(c)
		if err != nil {
			return err
		}

		if err := ioutil.WriteFile(filepath.Join(containersDir, id, configFileName), containerJSON, 0600); err != nil {
			return err
		}

		img, err := is.Get(imageID)
		if err != nil {
			return err
		}

		if err := ls.CreateRWLayerByGraphID(id, id, img.RootFS.ChainID()); err != nil {
			logrus.Errorf("migrate container error: %v", err)
			continue
		}

		logrus.Infof("migrated container %s to point to %s", id, imageID)

	}
	return nil
}

type refAdder interface {
	AddTag(ref reference.Named, id digest.Digest, force bool) error
	AddDigest(ref reference.Canonical, id digest.Digest, force bool) error
}

func migrateRefs(root, driverName string, rs refAdder, mappings map[string]image.ID) error {
	migrationFile := filepath.Join(root, migrationTagsFileName)
	if _, err := os.Lstat(migrationFile); !os.IsNotExist(err) {
		return err
	}

	type repositories struct {
		Repositories map[string]map[string]string
	}

	var repos repositories

	f, err := os.Open(filepath.Join(root, repositoriesFilePrefixLegacy+driverName))
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()
	if err := json.NewDecoder(f).Decode(&repos); err != nil {
		return err
	}

	for name, repo := range repos.Repositories {
		for tag, id := range repo {
			if strongID, exists := mappings[id]; exists {
				ref, err := reference.ParseNormalizedNamed(name)
				if err != nil {
					logrus.Errorf("migrate tags: invalid name %q, %q", name, err)
					continue
				}
				if !reference.IsNameOnly(ref) {
					logrus.Errorf("migrate tags: invalid name %q, unexpected tag or digest", name)
					continue
				}
				if dgst, err := digest.Parse(tag); err == nil {
					canonical, err := reference.WithDigest(reference.TrimNamed(ref), dgst)
					if err != nil {
						logrus.Errorf("migrate tags: invalid digest %q, %q", dgst, err)
						continue
					}
					if err := rs.AddDigest(canonical, strongID.Digest(), false); err != nil {
						logrus.Errorf("can't migrate digest %q for %q, err: %q", reference.FamiliarString(ref), strongID, err)
					}
				} else {
					tagRef, err := reference.WithTag(ref, tag)
					if err != nil {
						logrus.Errorf("migrate tags: invalid tag %q, %q", tag, err)
						continue
					}
					if err := rs.AddTag(tagRef, strongID.Digest(), false); err != nil {
						logrus.Errorf("can't migrate tag %q for %q, err: %q", reference.FamiliarString(ref), strongID, err)
					}
				}
				logrus.Infof("migrated tag %s:%s to point to %s", name, tag, strongID)
			}
		}
	}

	mf, err := os.Create(migrationFile)
	if err != nil {
		return err
	}
	mf.Close()

	return nil
}

func getParent(confDir string) (string, error) {
	jsonFile := filepath.Join(confDir, "json")
	imageJSON, err := ioutil.ReadFile(jsonFile)
	if err != nil {
		return "", err
	}
	var parent struct {
		Parent   string
		ParentID digest.Digest `json:"parent_id"`
	}
	if err := json.Unmarshal(imageJSON, &parent); err != nil {
		return "", err
	}
	if parent.Parent == "" && parent.ParentID != "" { // v1.9
		parent.Parent = parent.ParentID.Hex()
	}
	// compatibilityID for parent
	parentCompatibilityID, err := ioutil.ReadFile(filepath.Join(confDir, "parent"))
	if err == nil && len(parentCompatibilityID) > 0 {
		parent.Parent = string(parentCompatibilityID)
	}
	return parent.Parent, nil
}

func migrateImage(id, root string, ls graphIDRegistrar, is image.Store, ms metadata.Store, mappings map[string]image.ID) (err error) {
	defer func() {
		if err != nil {
			logrus.Errorf("migration failed for %v, err: %v", id, err)
		}
	}()

	parent, err := getParent(filepath.Join(root, graphDirName, id))
	if err != nil {
		return err
	}

	var parentID image.ID
	if parent != "" {
		var exists bool
		if parentID, exists = mappings[parent]; !exists {
			if err := migrateImage(parent, root, ls, is, ms, mappings); err != nil {
				// todo: fail or allow broken chains?
				return err
			}
			parentID = mappings[parent]
		}
	}

	rootFS := image.NewRootFS()
	var history []image.History

	if parentID != "" {
		parentImg, err := is.Get(parentID)
		if err != nil {
			return err
		}

		rootFS = parentImg.RootFS
		history = parentImg.History
	}

	diffIDData, err := ioutil.ReadFile(filepath.Join(root, graphDirName, id, migrationDiffIDFileName))
	if err != nil {
		return err
	}
	diffID, err := digest.Parse(string(diffIDData))
	if err != nil {
		return err
	}

	sizeStr, err := ioutil.ReadFile(filepath.Join(root, graphDirName, id, migrationSizeFileName))
	if err != nil {
		return err
	}
	size, err := strconv.ParseInt(string(sizeStr), 10, 64)
	if err != nil {
		return err
	}

	layer, err := ls.RegisterByGraphID(id, rootFS.ChainID(), layer.DiffID(diffID), filepath.Join(root, graphDirName, id, migrationTarDataFileName), size)
	if err != nil {
		return err
	}
	logrus.Infof("migrated layer %s to %s", id, layer.DiffID())

	jsonFile := filepath.Join(root, graphDirName, id, "json")
	imageJSON, err := ioutil.ReadFile(jsonFile)
	if err != nil {
		return err
	}

	h, err := imagev1.HistoryFromConfig(imageJSON, false)
	if err != nil {
		return err
	}
	history = append(history, h)

	rootFS.Append(layer.DiffID())

	config, err := imagev1.MakeConfigFromV1Config(imageJSON, rootFS, history)
	if err != nil {
		return err
	}
	strongID, err := is.Create(config)
	if err != nil {
		return err
	}
	logrus.Infof("migrated image %s to %s", id, strongID)

	if parentID != "" {
		if err := is.SetParent(strongID, parentID); err != nil {
			return err
		}
	}

	checksum, err := ioutil.ReadFile(filepath.Join(root, graphDirName, id, "checksum"))
	if err == nil { // best effort
		dgst, err := digest.Parse(string(checksum))
		if err == nil {
			V2MetadataService := metadata.NewV2MetadataService(ms)
			V2MetadataService.Add(layer.DiffID(), metadata.V2Metadata{Digest: dgst})
		}
	}
	_, err = ls.Release(layer)
	if err != nil {
		return err
	}

	mappings[id] = strongID
	return
}

func rawJSON(value interface{}) *json.RawMessage {
	jsonval, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	return (*json.RawMessage)(&jsonval)
}
