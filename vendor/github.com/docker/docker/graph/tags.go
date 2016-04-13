package graph

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/docker/distribution/digest"
	"github.com/docker/docker/daemon/events"
	"github.com/docker/docker/graph/tags"
	"github.com/docker/docker/image"
	"github.com/docker/docker/pkg/parsers"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/registry"
	"github.com/docker/docker/trust"
	"github.com/docker/docker/utils"
	"github.com/docker/libtrust"
)

const DEFAULTTAG = "latest"

type TagStore struct {
	path         string
	graph        *Graph
	Repositories map[string]Repository
	trustKey     libtrust.PrivateKey
	sync.Mutex
	// FIXME: move push/pull-related fields
	// to a helper type
	pullingPool     map[string]chan struct{}
	pushingPool     map[string]chan struct{}
	registryService *registry.Service
	eventsService   *events.Events
	trustService    *trust.TrustStore
}

type Repository map[string]string

// update Repository mapping with content of u
func (r Repository) Update(u Repository) {
	for k, v := range u {
		r[k] = v
	}
}

// return true if the contents of u Repository, are wholly contained in r Repository
func (r Repository) Contains(u Repository) bool {
	for k, v := range u {
		// if u's key is not present in r OR u's key is present, but not the same value
		if rv, ok := r[k]; !ok || (ok && rv != v) {
			return false
		}
	}
	return true
}

type TagStoreConfig struct {
	Graph    *Graph
	Key      libtrust.PrivateKey
	Registry *registry.Service
	Events   *events.Events
	Trust    *trust.TrustStore
}

func NewTagStore(path string, cfg *TagStoreConfig) (*TagStore, error) {
	abspath, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}

	store := &TagStore{
		path:            abspath,
		graph:           cfg.Graph,
		trustKey:        cfg.Key,
		Repositories:    make(map[string]Repository),
		pullingPool:     make(map[string]chan struct{}),
		pushingPool:     make(map[string]chan struct{}),
		registryService: cfg.Registry,
		eventsService:   cfg.Events,
		trustService:    cfg.Trust,
	}
	// Load the json file if it exists, otherwise create it.
	if err := store.reload(); os.IsNotExist(err) {
		if err := store.save(); err != nil {
			return nil, err
		}
	} else if err != nil {
		return nil, err
	}
	return store, nil
}

func (store *TagStore) save() error {
	// Store the json ball
	jsonData, err := json.Marshal(store)
	if err != nil {
		return err
	}
	if err := ioutil.WriteFile(store.path, jsonData, 0600); err != nil {
		return err
	}
	return nil
}

func (store *TagStore) reload() error {
	f, err := os.Open(store.path)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := json.NewDecoder(f).Decode(&store); err != nil {
		return err
	}
	return nil
}

func (store *TagStore) LookupImage(name string) (*image.Image, error) {
	// FIXME: standardize on returning nil when the image doesn't exist, and err for everything else
	// (so we can pass all errors here)
	repoName, ref := parsers.ParseRepositoryTag(name)
	if ref == "" {
		ref = DEFAULTTAG
	}
	var (
		err error
		img *image.Image
	)

	img, err = store.GetImage(repoName, ref)
	if err != nil {
		return nil, err
	}

	if img != nil {
		return img, err
	}

	// name must be an image ID.
	store.Lock()
	defer store.Unlock()
	if img, err = store.graph.Get(name); err != nil {
		return nil, err
	}

	return img, nil
}

// Return a reverse-lookup table of all the names which refer to each image
// Eg. {"43b5f19b10584": {"base:latest", "base:v1"}}
func (store *TagStore) ByID() map[string][]string {
	store.Lock()
	defer store.Unlock()
	byID := make(map[string][]string)
	for repoName, repository := range store.Repositories {
		for tag, id := range repository {
			name := utils.ImageReference(repoName, tag)
			if _, exists := byID[id]; !exists {
				byID[id] = []string{name}
			} else {
				byID[id] = append(byID[id], name)
				sort.Strings(byID[id])
			}
		}
	}
	return byID
}

func (store *TagStore) ImageName(id string) string {
	if names, exists := store.ByID()[id]; exists && len(names) > 0 {
		return names[0]
	}
	return stringid.TruncateID(id)
}

func (store *TagStore) DeleteAll(id string) error {
	names, exists := store.ByID()[id]
	if !exists || len(names) == 0 {
		return nil
	}
	for _, name := range names {
		if strings.Contains(name, ":") {
			nameParts := strings.Split(name, ":")
			if _, err := store.Delete(nameParts[0], nameParts[1]); err != nil {
				return err
			}
		} else {
			if _, err := store.Delete(name, ""); err != nil {
				return err
			}
		}
	}
	return nil
}

func (store *TagStore) Delete(repoName, ref string) (bool, error) {
	store.Lock()
	defer store.Unlock()
	deleted := false
	if err := store.reload(); err != nil {
		return false, err
	}

	repoName = registry.NormalizeLocalName(repoName)

	if ref == "" {
		// Delete the whole repository.
		delete(store.Repositories, repoName)
		return true, store.save()
	}

	repoRefs, exists := store.Repositories[repoName]
	if !exists {
		return false, fmt.Errorf("No such repository: %s", repoName)
	}

	if _, exists := repoRefs[ref]; exists {
		delete(repoRefs, ref)
		if len(repoRefs) == 0 {
			delete(store.Repositories, repoName)
		}
		deleted = true
	}

	return deleted, store.save()
}

func (store *TagStore) Tag(repoName, tag, imageName string, force bool) error {
	return store.SetLoad(repoName, tag, imageName, force, nil)
}

func (store *TagStore) SetLoad(repoName, tag, imageName string, force bool, out io.Writer) error {
	img, err := store.LookupImage(imageName)
	store.Lock()
	defer store.Unlock()
	if err != nil {
		return err
	}
	if tag == "" {
		tag = tags.DEFAULTTAG
	}
	if err := validateRepoName(repoName); err != nil {
		return err
	}
	if err := tags.ValidateTagName(tag); err != nil {
		if _, formatError := err.(tags.ErrTagInvalidFormat); !formatError {
			return err
		}
		if _, dErr := digest.ParseDigest(tag); dErr != nil {
			// Still return the tag validation error.
			// It's more likely to be a user generated issue.
			return err
		}
	}
	if err := store.reload(); err != nil {
		return err
	}
	var repo Repository
	repoName = registry.NormalizeLocalName(repoName)
	if r, exists := store.Repositories[repoName]; exists {
		repo = r
		if old, exists := store.Repositories[repoName][tag]; exists {

			if !force {
				return fmt.Errorf("Conflict: Tag %s is already set to image %s, if you want to replace it, please use -f option", tag, old)
			}

			if old != img.ID && out != nil {

				fmt.Fprintf(out, "The image %s:%s already exists, renaming the old one with ID %s to empty string\n", repoName, tag, old[:12])

			}
		}
	} else {
		repo = make(map[string]string)
		store.Repositories[repoName] = repo
	}
	repo[tag] = img.ID
	return store.save()
}

// SetDigest creates a digest reference to an image ID.
func (store *TagStore) SetDigest(repoName, digest, imageName string) error {
	img, err := store.LookupImage(imageName)
	if err != nil {
		return err
	}

	if err := validateRepoName(repoName); err != nil {
		return err
	}

	if err := validateDigest(digest); err != nil {
		return err
	}

	store.Lock()
	defer store.Unlock()
	if err := store.reload(); err != nil {
		return err
	}

	repoName = registry.NormalizeLocalName(repoName)
	repoRefs, exists := store.Repositories[repoName]
	if !exists {
		repoRefs = Repository{}
		store.Repositories[repoName] = repoRefs
	} else if oldID, exists := repoRefs[digest]; exists && oldID != img.ID {
		return fmt.Errorf("Conflict: Digest %s is already set to image %s", digest, oldID)
	}

	repoRefs[digest] = img.ID
	return store.save()
}

func (store *TagStore) Get(repoName string) (Repository, error) {
	store.Lock()
	defer store.Unlock()
	if err := store.reload(); err != nil {
		return nil, err
	}
	repoName = registry.NormalizeLocalName(repoName)
	if r, exists := store.Repositories[repoName]; exists {
		return r, nil
	}
	return nil, nil
}

func (store *TagStore) GetImage(repoName, refOrID string) (*image.Image, error) {
	repo, err := store.Get(repoName)

	if err != nil {
		return nil, err
	}
	if repo == nil {
		return nil, nil
	}

	store.Lock()
	defer store.Unlock()
	if imgID, exists := repo[refOrID]; exists {
		return store.graph.Get(imgID)
	}

	// If no matching tag is found, search through images for a matching image id
	// iff it looks like a short ID or would look like a short ID
	if stringid.IsShortID(stringid.TruncateID(refOrID)) {
		for _, revision := range repo {
			if strings.HasPrefix(revision, refOrID) {
				return store.graph.Get(revision)
			}
		}
	}

	return nil, nil
}

func (store *TagStore) GetRepoRefs() map[string][]string {
	store.Lock()
	reporefs := make(map[string][]string)

	for name, repository := range store.Repositories {
		for tag, id := range repository {
			shortID := stringid.TruncateID(id)
			reporefs[shortID] = append(reporefs[shortID], utils.ImageReference(name, tag))
		}
	}
	store.Unlock()
	return reporefs
}

// Validate the name of a repository
func validateRepoName(name string) error {
	if name == "" {
		return fmt.Errorf("Repository name can't be empty")
	}
	if name == "scratch" {
		return fmt.Errorf("'scratch' is a reserved name")
	}
	return nil
}

func validateDigest(dgst string) error {
	if dgst == "" {
		return errors.New("digest can't be empty")
	}
	if _, err := digest.ParseDigest(dgst); err != nil {
		return err
	}
	return nil
}

func (store *TagStore) poolAdd(kind, key string) (chan struct{}, error) {
	store.Lock()
	defer store.Unlock()

	if c, exists := store.pullingPool[key]; exists {
		return c, fmt.Errorf("pull %s is already in progress", key)
	}
	if c, exists := store.pushingPool[key]; exists {
		return c, fmt.Errorf("push %s is already in progress", key)
	}

	c := make(chan struct{})
	switch kind {
	case "pull":
		store.pullingPool[key] = c
	case "push":
		store.pushingPool[key] = c
	default:
		return nil, fmt.Errorf("Unknown pool type")
	}
	return c, nil
}

func (store *TagStore) poolRemove(kind, key string) error {
	store.Lock()
	defer store.Unlock()
	switch kind {
	case "pull":
		if c, exists := store.pullingPool[key]; exists {
			close(c)
			delete(store.pullingPool, key)
		}
	case "push":
		if c, exists := store.pushingPool[key]; exists {
			close(c)
			delete(store.pushingPool, key)
		}
	default:
		return fmt.Errorf("Unknown pool type")
	}
	return nil
}
