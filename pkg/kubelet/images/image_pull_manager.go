/*
Copyright 2024 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package images

import (
	"context"
	"crypto/sha256"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/credentialprovider"
	imagemanagerv1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/config/imagemanager/v1alpha1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util/parsers"
)

var _ ImagePullManager = &FileBasedImagePullManager{}

// NamedLockSet stores named locks in order to allow to partition context access
// such that callers that are mutually exclusive based on a string value can
// access the same context at the same time, compared to a global lock that
// would create an unnecessary bottleneck.
type NamedLockSet struct {
	globalLock sync.Mutex
	locks      map[string]*sync.Mutex
}

func NewNamedLockSet() *NamedLockSet {
	return &NamedLockSet{
		globalLock: sync.Mutex{},
		locks:      map[string]*sync.Mutex{},
	}
}

func (n *NamedLockSet) Lock(name string) {
	func() {
		n.globalLock.Lock()
		defer n.globalLock.Unlock()
		if _, ok := n.locks[name]; !ok {
			n.locks[name] = &sync.Mutex{}
		}
	}()
	// This call cannot be guarded by the global lock as it would block the access
	// to the other locks
	n.locks[name].Lock()
}

// Unlock unlocks the named lock. Can only be called after a previous Lock() call
// for the same named lock.
func (n *NamedLockSet) Unlock(name string) {
	// cannot be locked by the global lock as it would deadlock once GlobalLock() gets activated
	if _, ok := n.locks[name]; ok {
		n.locks[name].Unlock()
	}
}

// GlobalLock first locks access to the named locks and then locks all of the
// set locks
func (n *NamedLockSet) GlobalLock() {
	n.globalLock.Lock()
	for _, l := range n.locks {
		l.Lock()
	}
}

// GlobalUnlock should only be called after GlobalLock(). It unlocks all the locks
// of the set and then it also unlocks the global lock gating access to the set locks.
func (n *NamedLockSet) GlobalUnlock() {
	for _, l := range n.locks {
		l.Unlock()
	}
	n.globalLock.Unlock()
}

// FileBasedImagePullManager is an implementation of the ImagePullManager. It
// tracks images pulled by the kubelet by creating records about ongoing and
// successful pulls.
// It tracks the credentials used with each successful pull in order to be able
// to distinguish tenants requesting access to an image that exists on the kubelet's
// node.
type FileBasedImagePullManager struct {
	pullingDir string
	pulledDir  string

	imagePullPolicy ImagePullPolicyEnforcer

	imageService kubecontainer.ImageService

	intentAccessors *NamedLockSet  // image -> sync.Mutex
	intentCounters  map[string]int // image -> number of current in-flight pulls

	pulledAccessors *NamedLockSet // imageRef -> sync.Mutex
}

func NewFileBasedImagePullManager(ctx context.Context, kubeletDir string, imagePullPolicy ImagePullPolicyEnforcer, imageService kubecontainer.ImageService) (*FileBasedImagePullManager, error) {
	m := &FileBasedImagePullManager{
		pullingDir: filepath.Join(kubeletDir, "image_manager", "pulling"),
		pulledDir:  filepath.Join(kubeletDir, "image_manager", "pulled"),

		imagePullPolicy: imagePullPolicy,

		imageService: imageService,

		intentAccessors: NewNamedLockSet(),
		intentCounters:  make(map[string]int),

		pulledAccessors: NewNamedLockSet(),
	}

	if err := os.MkdirAll(m.pullingDir, 0700); err != nil {
		return nil, err
	}

	if err := os.MkdirAll(m.pulledDir, 0700); err != nil {
		return nil, err
	}

	m.startupHousekeeping(ctx)

	return m, nil
}

func (f *FileBasedImagePullManager) RecordPullIntent(image string) error {
	f.intentAccessors.Lock(image)
	defer f.intentAccessors.Unlock(image)

	intent := imagemanagerv1alpha1.ImagePullIntent{
		Image: image,
	}
	intentBytes, err := json.Marshal(intent)
	if err != nil {
		return err
	}

	if err := writeFile(f.pullingDir, cacheFilename(image), intentBytes); err != nil {
		return err
	}

	f.intentCounters[image]++
	return nil
}

func (f *FileBasedImagePullManager) RecordImagePulled(image, imageRef string, credentials *imagemanagerv1alpha1.ImagePullCredentials) {
	if err := f.writePulledRecord(image, imageRef, credentials); err != nil {
		klog.ErrorS(err, "failed to write image pulled record", "imageRef", imageRef)
		return
	}

	f.dereferenceImagePullIntent(image)
}

// writePulledRecord writes an ImagePulledRecord into the f.pulledDir directory.
func (f *FileBasedImagePullManager) writePulledRecord(image, imageRef string, credentials *imagemanagerv1alpha1.ImagePullCredentials) error {
	f.pulledAccessors.Lock(imageRef)
	defer f.pulledAccessors.Unlock(imageRef)

	var pulledRecord imagemanagerv1alpha1.ImagePulledRecord
	var pulledRecordChanged bool

	sanitizedImage, err := trimImageTagDigest(image)
	if err != nil {
		return fmt.Errorf("invalidate image name %q: %w", image, err)
	}

	pulledRecordPath := filepath.Join(f.pulledDir, cacheFilename(imageRef))
	pulledFile, err := os.ReadFile(pulledRecordPath)
	if os.IsNotExist(err) {
		pulledRecordChanged = true
		pulledRecord = imagemanagerv1alpha1.ImagePulledRecord{
			LastUpdatedTime:   metav1.Time{Time: time.Now()},
			ImageRef:          imageRef,
			CredentialMapping: make(map[string]imagemanagerv1alpha1.ImagePullCredentials),
		}
		if credentials != nil {
			pulledRecord.CredentialMapping[sanitizedImage] = *credentials
		}
	} else if err == nil {
		if err := json.Unmarshal(pulledFile, &pulledRecord); err != nil {
			return fmt.Errorf("failed to decode ImagePulledRecord: %w", err)
		}

		pulledRecord, pulledRecordChanged = pulledRecordMergeNewCreds(&pulledRecord, sanitizedImage, credentials)
	} else {
		return fmt.Errorf("failed to open %q for reading: %w", pulledRecordPath, err)
	}

	if !pulledRecordChanged {
		return nil
	}
	recordBytes, err := json.Marshal(pulledRecord)
	if err != nil {
		return fmt.Errorf("failed to serialize ImagePulledRecord: %w", err)
	}

	if err := writeFile(f.pulledDir, cacheFilename(imageRef), recordBytes); err != nil {
		return fmt.Errorf("failed to write ImagePulledRecord file: %w", err)
	}

	return nil

}

func (f *FileBasedImagePullManager) RecordImagePullFailed(image string) {
	f.dereferenceImagePullIntent(image)
}

// dereferenceImagePullIntent decreses the number of how many times image pull
// intent for a given `image` was requested, and removes the ImagePullIntent file
// if the reference counter for the image reaches zero.
func (f *FileBasedImagePullManager) dereferenceImagePullIntent(image string) {
	f.intentAccessors.Lock(image)
	defer f.intentAccessors.Unlock(image)

	f.intentCounters[image]--
	if f.intentCounters[image] > 0 {
		return
	}
	f.intentCounters[image] = 0 // safeguard, we should never actually go below zero

	intentRecordPath := filepath.Join(f.pullingDir, cacheFilename(image))
	if err := os.Remove(intentRecordPath); err != nil && !os.IsNotExist(err) {
		klog.ErrorS(err, "failed to remove file", "filePath", intentRecordPath)
	}
}

// TODO: if we salt the hashes, we must pass the creds in cleartext here
func (f *FileBasedImagePullManager) MustAttemptImagePull(image, imageRef string, podSecrets []imagemanagerv1alpha1.ImagePullSecret) bool {
	if len(imageRef) == 0 {
		return true
	}

	pulledFilePath := filepath.Join(f.pulledDir, cacheFilename(imageRef))

	var imageRecordExists bool
	var pulledFile []byte

	err := func() error {
		// don't allow changes to the files we're using for our decision
		f.pulledAccessors.Lock(imageRef)
		defer f.pulledAccessors.Unlock(imageRef)
		f.intentAccessors.Lock(image)
		defer f.intentAccessors.Unlock(image)

		var err error
		pulledFile, err = os.ReadFile(pulledFilePath)
		if err == nil {
			imageRecordExists = true
		} else if !os.IsNotExist(err) {
			return fmt.Errorf("failed to open %q for reading: %w", pulledFilePath, err)
		} else {
			pullIntentPath := filepath.Join(f.pullingDir, cacheFilename(image))
			_, err := os.ReadFile(pullIntentPath)
			if err == nil {
				imageRecordExists = true
			} else if !os.IsNotExist(err) {
				return fmt.Errorf("failed to open %q for reading: %w", pullIntentPath, err)

			}
		}
		return nil
	}()

	if err != nil {
		klog.ErrorS(err, "Unable to read files in directories containing records about image pulls")
		return true
	}

	if !f.imagePullPolicy(image, len(imageRef) > 0, imageRecordExists) {
		return false
	}

	if len(pulledFile) == 0 {
		// we have no proper records of the image being pulled in the past, we can short-circuit here
		return true
	}

	pulledDeserialized := imagemanagerv1alpha1.ImagePulledRecord{}
	if err := json.Unmarshal(pulledFile, &pulledDeserialized); err != nil {
		klog.ErrorS(err, "failed to decode ImagePulledRecord")
		return true
	}

	sanitizedImage, err := trimImageTagDigest(image)
	if err != nil {
		klog.ErrorS(err, "failed to parse image name, forcing image credentials reverification", "image", sanitizedImage)
		return true
	}

	cachedCreds, ok := pulledDeserialized.CredentialMapping[sanitizedImage]
	if !ok {
		return true
	}

	if cachedCreds.NodePodsAccessible {
		// anyone on this node can access the image
		return false
	}

	if len(cachedCreds.KubernetesSecrets) == 0 {
		return true
	}

	for _, podSecret := range podSecrets {
		for _, cachedSecret := range cachedCreds.KubernetesSecrets {

			if podSecret.CredentialHash == cachedSecret.CredentialHash { // TODO: if we're salting, we need to rehash the podSecret with the cachedSecret salt
				// TODO: should we record the new secret in case it has different coordinates? If the secret rotates, we will pull unnecessarily otherwise
				return false
			}

			if podSecret.UID == cachedSecret.UID &&
				podSecret.Namespace == cachedSecret.Namespace &&
				podSecret.Name == cachedSecret.Name {
				// TODO: should we record the new creds in this case so that we don't pull if these are present in a different secret?
				return false
			}
		}
	}

	return true
}

func (f *FileBasedImagePullManager) PruneUnknownRecords(imageList []string, until time.Time) {
	f.pulledAccessors.GlobalLock()
	defer f.pulledAccessors.GlobalUnlock()

	var imageRecords []string
	err := filepath.WalkDir(f.pulledDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if path == f.pulledDir {
			return nil
		}

		if d.IsDir() {
			klog.V(5).InfoS("path is a directory, skipping", "path", path)
			return nil
		}

		fileContent, err := os.ReadFile(path)
		if err != nil {
			klog.V(5).InfoS("failed to read file, skipping", "filePath", path, "error", err)
			return nil
		}

		pullRecord := imagemanagerv1alpha1.ImagePulledRecord{}
		if err := json.Unmarshal(fileContent, &pullRecord); err != nil {
			klog.V(5).InfoS("failed to deserialize, skipping file", "filePath", path, "error", err)
			return nil
		}

		if pullRecord.LastUpdatedTime.Time.Before(until) {
			imageRecords = append(imageRecords, pullRecord.ImageRef)
		}
		return nil
	})

	if err != nil {
		klog.ErrorS(err, "failed to garbage collect ImagePulledRecord files")
		return
	}

	imagesInUse := sets.New(imageList...)
	for _, ir := range imageRecords {
		if imagesInUse.Has(ir) {
			continue
		}

		recordToRemovePath := filepath.Join(f.pulledDir, cacheFilename(ir))
		if err := os.Remove(recordToRemovePath); err != nil {
			klog.ErrorS(err, "failed to remove an ImagePulledRecord file", "filePath", recordToRemovePath)
		}
	}

}

// startupHousekeeping gathers all the images from pull intent records that exist
// from the previous kubelet runs.
// If the CRI reports any of the above images as already pulled, we turn the
// pull intent into a pulled record and the original pull intent is deleted.
func (f *FileBasedImagePullManager) startupHousekeeping(ctx context.Context) {
	inFlightPulls := make(map[string]string) // image -> path

	f.pulledAccessors.GlobalLock()
	defer f.pulledAccessors.GlobalUnlock()

	// walk the pulling directory for any pull intent records
	err := filepath.WalkDir(f.pullingDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if path == f.pulledDir {
			return nil
		}

		if d.IsDir() {
			klog.V(4).InfoS("path is a directory, skipping", "path", path)
			return nil
		}

		fileContent, err := os.ReadFile(path)
		if err != nil {
			klog.ErrorS(err, "skipping file, failed to read", "filePath", path)
			return nil
		}

		var intent imagemanagerv1alpha1.ImagePullIntent
		if err := json.Unmarshal(fileContent, &intent); err != nil {
			klog.ErrorS(err, "skipping file, failed to deserialize to ImagePullIntent", "filePath", path)
			return nil
		}

		inFlightPulls[intent.Image] = path
		return nil
	})
	if err != nil {
		klog.ErrorS(err, "there was an error walking the directory", "directoryPath", f.pullingDir)
		return
	}

	if len(inFlightPulls) == 0 {
		return
	}

	images, err := f.imageService.ListImages(ctx)
	if err != nil {
		klog.ErrorS(err, "failed to list images")
	}

	// Each of the images known to the CRI might consist of multiple tags and digests,
	// which is what we track in the ImagePullIntent - we need to go through all of these
	// for each image.
	for _, image := range images {
		imageName := searchForExistingTagDigest(inFlightPulls, image)
		if len(imageName) == 0 {
			continue
		}

		if err := f.writePulledRecord(imageName, image.ID, nil); err != nil {
			klog.ErrorS(err, "failed to write an image pull record", "imageRef", image.ID)
			continue
		}
		removePath := inFlightPulls[imageName]
		if err := os.Remove(removePath); err != nil {
			klog.V(2).InfoS("failed to remove image pull intent file", "filePath", removePath, "error", err)
		}

	}
}

var _ ImagePullManager = &NoopImagePullManager{}

type NoopImagePullManager struct{}

func (m *NoopImagePullManager) RecordPullIntent(_ string) error { return nil }
func (m *NoopImagePullManager) RecordImagePulled(_, _ string, _ *imagemanagerv1alpha1.ImagePullCredentials) {
}
func (m *NoopImagePullManager) RecordImagePullFailed(image string) {}
func (m *NoopImagePullManager) MustAttemptImagePull(_, _ string, _ []imagemanagerv1alpha1.ImagePullSecret) bool {
	return false
}
func (m *NoopImagePullManager) PruneUnknownRecords(_ []string, _ time.Time) {}

// searchForExistingTagDigest loop through the `image` RepoDigests and RepoTags
// and tries to find a digest/tag in `trackedImages`, which is a map of
// containerImage -> pulling intent path.
func searchForExistingTagDigest(trackedImages map[string]string, image kubecontainer.Image) string {
	for _, digest := range image.RepoDigests {
		if _, ok := trackedImages[digest]; ok {
			return digest
		}
	}

	for _, tag := range image.RepoTags {
		if _, ok := trackedImages[tag]; ok {
			return tag
		}
	}

	return ""
}

type kubSecretCoordinates struct {
	UID       string
	Namespace string
	Name      string
}

// pulledRecordMergeNewCreds merges the credentials from `newCreds` into the `orig`
// record for the `imageNoTagDigest` image.
// `imageNoTagDigest` is the content of the `image` field from a pod's container
// after any tag or digest were removed from it.
func pulledRecordMergeNewCreds(orig *imagemanagerv1alpha1.ImagePulledRecord, imageNoTagDigest string, newCreds *imagemanagerv1alpha1.ImagePullCredentials) (imagemanagerv1alpha1.ImagePulledRecord, bool) {
	ret := orig.DeepCopy()
	if newCreds == nil {
		return *ret, false
	}
	selectedCreds, found := ret.CredentialMapping[imageNoTagDigest]
	if !found {
		if ret.CredentialMapping == nil {
			ret.CredentialMapping = make(map[string]imagemanagerv1alpha1.ImagePullCredentials)
		}
		ret.CredentialMapping[imageNoTagDigest] = *newCreds
	} else {
		if newCreds.NodePodsAccessible {
			selectedCreds.NodePodsAccessible = true
			selectedCreds.KubernetesSecrets = nil
		} else {
			selectedCreds.KubernetesSecrets = mergePullSecrets(selectedCreds.KubernetesSecrets, newCreds.KubernetesSecrets)
		}
		ret.CredentialMapping[imageNoTagDigest] = selectedCreds
	}

	updated := !reflect.DeepEqual(orig.CredentialMapping, ret.CredentialMapping)
	if updated {
		ret.LastUpdatedTime = metav1.Time{Time: time.Now()}
	}
	return *ret, updated
}

// mergePullSecrets merges two slices of ImagePullSecret object into one while
// keeping the objects unique per `Namespace, Name, UID` key.
//
// In case an object from the `new` slice has the same `Namespace, Name, UID` combination
// as an object from `orig`, the result will use the CredentialHash value of the
// object from `new`.
//
// The returned slice is sorted by Namespace, Name and UID (in this order).
func mergePullSecrets(orig, new []imagemanagerv1alpha1.ImagePullSecret) []imagemanagerv1alpha1.ImagePullSecret {
	credSet := make(map[kubSecretCoordinates]string)
	for _, secret := range orig {
		credSet[kubSecretCoordinates{
			UID:       secret.UID,
			Namespace: secret.Namespace,
			Name:      secret.Name,
		}] = secret.CredentialHash
	}

	for _, s := range new {
		credSet[kubSecretCoordinates{UID: s.UID, Namespace: s.Namespace, Name: s.Name}] = s.CredentialHash
	}

	ret := make([]imagemanagerv1alpha1.ImagePullSecret, 0, len(credSet))
	for coords, hash := range credSet {
		ret = append(ret, imagemanagerv1alpha1.ImagePullSecret{
			UID:            coords.UID,
			Namespace:      coords.Namespace,
			Name:           coords.Name,
			CredentialHash: hash,
		})
	}
	// we don't need to use the stable version because secret coordinates used for ordering are unique in the set
	slices.SortFunc(ret, imagePullSecretLess)

	return ret
}

func cacheFilename(image string) string {
	return fmt.Sprintf("sha256-%x", sha256.Sum256([]byte(image)))
}

// imagePullSecretLess is a helper function to define ordering in a slice of
// ImagePullSecret objects.
func imagePullSecretLess(a, b imagemanagerv1alpha1.ImagePullSecret) int {
	if a.Namespace < b.Namespace {
		return -1
	} else if a.Namespace > b.Namespace {
		return 1
	}

	if a.Name < b.Name {
		return -1
	} else if a.Name > b.Name {
		return 1
	}

	if a.UID < b.UID {
		return -1
	}
	return 1
}

// TODO: should we salt the hashes to prevent on-disk introspection for identical secrets?
func hashImageCredentials(creds credentialprovider.AuthConfig) (string, error) {
	credBytes, err := json.Marshal(creds)
	if err != nil {
		return "", err
	}

	hash := sha256.New()
	hash.Write([]byte(credBytes))
	// hash.Write([]byte(salt))
	return fmt.Sprintf("%x", hash.Sum(nil)), nil
}

// writeFile writes `content` to the file with name `filename` in directory `dir`
func writeFile(dir, filename string, content []byte) error {
	// create target folder if it does not exists yet
	if err := os.MkdirAll(dir, 0700); err != nil {
		return fmt.Errorf("failed to create directory %q: %w", dir, err)
	}

	filePath := filepath.Join(dir, filename)
	return os.WriteFile(filePath, content, 0600)
}

// trimImageTagDigest removes the tag and digest from an image name
func trimImageTagDigest(containerImage string) (string, error) {
	imageName, _, _, err := parsers.ParseImageName(containerImage)
	return imageName, err
}
