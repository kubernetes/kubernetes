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
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/credentialprovider"
	imagemanagerv1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/config/imagemanager/v1alpha1"
	"k8s.io/kubernetes/pkg/util/parsers"
)

var _ ImagePullManager = &FileBasedImagePullManager{}

type NamedLockSet struct {
	locks sync.Map
}

func NewNamedLockSet() *NamedLockSet {
	return &NamedLockSet{
		locks: sync.Map{},
	}
}

func (n *NamedLockSet) Lock(name string) {
	lock, _ := n.locks.LoadOrStore(name, &sync.Mutex{})
	lock.(*sync.Mutex).Lock()
}

func (n *NamedLockSet) Unlock(name string) {
	lock, _ := n.locks.Load(name)
	lock.(*sync.Mutex).Unlock()
}

type FileBasedImagePullManager struct {
	pullingDir string
	pulledDir  string

	imagePullPolicy ImagePullPolicyEnforcer

	intentAccessors *NamedLockSet     // image -> sync.Mutex
	intentCounters  map[string]uint16 // image -> number of current in-flight pulls

	pulledAccessors *NamedLockSet // imageRef -> sync.Mutex
}

func NewFileBasedImagePullManager(kubeletDir string, imagePullPolicy ImagePullPolicyEnforcer) (*FileBasedImagePullManager, error) {
	m := &FileBasedImagePullManager{
		pullingDir: filepath.Join(kubeletDir, "image_manager", "pulling"),
		pulledDir:  filepath.Join(kubeletDir, "image_manager", "pulled"),

		imagePullPolicy: imagePullPolicy,

		intentAccessors: NewNamedLockSet(),
		intentCounters:  make(map[string]uint16),

		pulledAccessors: NewNamedLockSet(),
	}

	if err := os.MkdirAll(m.pullingDir, 0700); err != nil {
		return nil, err
	}

	if err := os.MkdirAll(m.pulledDir, 0700); err != nil {
		return nil, err
	}

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
		klog.Errorf("failed to write pulled record: %v", err.Error())
		return
	}

	// FIXME: copy pasta code from RecordPullIntent
	f.intentAccessors.Lock(image)
	defer f.intentAccessors.Unlock(image)

	f.intentCounters[image]--
	if f.intentCounters[image] != 0 {
		return
	}

	intentRecordPath := filepath.Join(f.pullingDir, cacheFilename(image))
	if err := os.Remove(intentRecordPath); err != nil {
		klog.Errorf("failed to remove file %s: %v", intentRecordPath, err)
	}
}

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
			LastUpdatedTime: metav1.Time{Time: time.Now()},
			ImageRef:        imageRef,
			CredentialMapping: map[string]imagemanagerv1alpha1.ImagePullCredentials{
				sanitizedImage: *credentials,
			},
		}
	} else if err == nil {
		if err := json.Unmarshal(pulledFile, &pulledRecord); err != nil {
			return fmt.Errorf("failed to decode ImagePulledRecord: %w", err)
		}

		pulledRecord, pulledRecordChanged = pulledRecordMergeNewCreds(&pulledRecord, image, credentials)
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
	f.intentAccessors.Lock(image)
	defer f.intentAccessors.Unlock(image)

	f.intentCounters[image]--
	if f.intentCounters[image] != 0 {
		return
	}

	intentRecordPath := filepath.Join(f.pullingDir, cacheFilename(image))
	if err := os.Remove(intentRecordPath); err != nil {
		klog.Errorf("failed to remove file %s: %v", intentRecordPath, err)
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
		klog.Errorf(err.Error())
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
		klog.Errorf("failed to decode ImagePulledRecord: %v", err)
		return true
	}

	sanitizedImage, err := trimImageTagDigest(image)
	if err != nil {
		klog.Errorf("failed to parse image name: %q - forcing image credentials reverification; parsing error %v", sanitizedImage, err)
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
	// TODO: also cleanup the lock maps for intent/pull records?
	panic("implement me")
}

type kubSecretCoordinates struct {
	UID       string
	Namespace string
	Name      string
}

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
