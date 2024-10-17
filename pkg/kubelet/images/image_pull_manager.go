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
	"path"
	"reflect"
	"slices"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/credentialprovider"
	imagemanagerv1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/config/imagemanager/v1alpha1"
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
		pullingDir: path.Join(kubeletDir, "image_manager", "pulling"),
		pulledDir:  path.Join(kubeletDir, "image_manager", "pulled"),

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

func (f *FileBasedImagePullManager) RecordPullIntent(image string) error { // TODO: should probably be locked, otherwise the file can be deleted right after it's written by someone else and the intent counter would stay >0
	f.intentAccessors.Lock(image)
	defer f.intentAccessors.Unlock(image)

	intent := imagemanagerv1alpha1.ImagePullIntent{
		Image: image,
	}
	intentBytes, err := json.Marshal(intent)
	if err != nil {
		return err
	}

	// TODO: always make sure the directory tree exists
	if err := os.WriteFile(cacheFilePath(f.pullingDir, image), intentBytes, 0600); err != nil {
		return err
	}

	f.intentCounters[image]++
	return nil
}

func (f *FileBasedImagePullManager) RecordImagePulled(image, imageRef string, credentials *imagemanagerv1alpha1.ImagePullCredentials) {
	func() { // TODO: separate unexported function?
		f.pulledAccessors.Lock(imageRef)
		defer f.pulledAccessors.Unlock(imageRef)

		// FIXME: the code below appears messy
		pulledFile, err := os.ReadFile(cacheFilePath(f.pulledDir, imageRef))

		var pulledRecord imagemanagerv1alpha1.ImagePulledRecord
		var pulledRecordChanged bool
		if os.IsNotExist(err) {
			pulledRecordChanged = true
			pulledRecord = imagemanagerv1alpha1.ImagePulledRecord{
				LastUpdatedTime: metav1.Time{Time: time.Now()},
				ImageRef:        imageRef,
				CredentialMapping: map[string]imagemanagerv1alpha1.ImagePullCredentials{
					image: *credentials, // TODO: double-check `image` is the key
				},
			}
		} else if err == nil {
			pulleRecord := imagemanagerv1alpha1.ImagePulledRecord{}
			if err := json.Unmarshal(pulledFile, &pulleRecord); err != nil {
				klog.Errorf("failed to decode ImagePulledRecord: %v", err)
				return
			}

			pulledRecord, pulledRecordChanged = pulledRecordMergeNewCreds(&pulledRecord, image, credentials)
		} else {
			klog.Errorf("failed to open %q for reading: %v", cacheFilePath(f.pulledDir, imageRef), err)
			return
		}

		if pulledRecordChanged {
			recordBytes, err := json.Marshal(pulledRecord)
			if err != nil {
				klog.Errorf("failed to serialize ImagePulledRecord: %v", err)
				return
			}

			// TODO: always make sure the directory tree exists
			if err := os.WriteFile(cacheFilePath(f.pulledDir, imageRef), recordBytes, 0600); err != nil {
				klog.Errorf("failed to write ImagePulledRecord file: %v", err)
				return
			}
		}
	}()

	// FIXME: copy pasta code from RecordPullIntent
	f.intentAccessors.Lock(image)
	defer f.intentAccessors.Unlock(image)

	f.intentCounters[image]--
	if f.intentCounters[image] != 0 {
		return
	}

	if err := os.Remove(cacheFilePath(f.pullingDir, image)); err != nil {
		klog.Errorf("failed to remove file %s: %v", cacheFilePath(f.pullingDir, image), err)
	}
}

func (f *FileBasedImagePullManager) RecordImagePullFailed(image string) {
	f.intentAccessors.Lock(image)
	defer f.intentAccessors.Unlock(image)

	f.intentCounters[image]--
	if f.intentCounters[image] != 0 {
		return
	}

	if err := os.Remove(cacheFilePath(f.pullingDir, image)); err != nil {
		klog.Errorf("failed to remove file %s: %v", cacheFilePath(f.pullingDir, image), err)
	}
}

// TODO: if we salt the hashes, we must pass the creds in cleartext here
func (f *FileBasedImagePullManager) MustAttemptImagePull(image, imageRef string, podSecrets []imagemanagerv1alpha1.ImagePullSecret) bool {
	if len(imageRef) == 0 {
		return true
	}

	f.pulledAccessors.Lock(imageRef)
	defer f.pulledAccessors.Unlock(imageRef)
	f.intentAccessors.Lock(image)
	defer f.intentAccessors.Unlock(image)

	pulledFilePath := cacheFilePath(f.pulledDir, imageRef)

	var imageRecordExists bool
	pulledFile, err := os.ReadFile(pulledFilePath)
	if err == nil {
		imageRecordExists = true
	} else if !os.IsNotExist(err) {
		klog.Errorf("failed to open %q for reading: %v", pulledFilePath, err)
		return true
	} else { // TODO: the below is quite messy?
		pullIntentPath := cacheFilePath(f.pullingDir, image)
		_, err := os.ReadFile(pullIntentPath)
		if err == nil {
			imageRecordExists = true
		} else if !os.IsNotExist(err) {
			klog.Errorf("failed to open %q for reading: %v", pullIntentPath, err)
			return true
		}
	}
	// TODO: we can actually probably unlock everything here

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

	cachedCreds, ok := pulledDeserialized.CredentialMapping[image]
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

func pulledRecordMergeNewCreds(orig *imagemanagerv1alpha1.ImagePulledRecord, image string, newCreds *imagemanagerv1alpha1.ImagePullCredentials) (imagemanagerv1alpha1.ImagePulledRecord, bool) {
	ret := orig.DeepCopy()
	selectedCreds, found := ret.CredentialMapping[image]
	if !found {
		if ret.CredentialMapping == nil {
			ret.CredentialMapping = make(map[string]imagemanagerv1alpha1.ImagePullCredentials)
		}
		ret.CredentialMapping[image] = *newCreds
	} else {
		if newCreds.NodePodsAccessible {
			selectedCreds.NodePodsAccessible = true
			selectedCreds.KubernetesSecrets = nil
		} else { // FIXME: probably worth its own function
			credSet := make(map[kubSecretCoordinates]string)
			for _, secret := range selectedCreds.KubernetesSecrets {
				credSet[kubSecretCoordinates{
					UID:       secret.UID,
					Namespace: secret.Namespace,
					Name:      secret.Name,
				}] = secret.CredentialHash
			}

			for _, s := range newCreds.KubernetesSecrets {
				credSet[kubSecretCoordinates{UID: s.UID, Namespace: s.Namespace, Name: s.Name}] = s.CredentialHash
			}

			newSecrets := make([]imagemanagerv1alpha1.ImagePullSecret, 0, len(credSet))
			for coords, hash := range credSet {
				newSecrets = append(newSecrets, imagemanagerv1alpha1.ImagePullSecret{
					UID:            coords.UID,
					Namespace:      coords.Namespace,
					Name:           coords.Name,
					CredentialHash: hash,
				})
			}
			// we don't need to use the stable version because secret coordinates used for ordering are unique in the set
			slices.SortFunc(newSecrets, imagePullSecretLess)
			selectedCreds.KubernetesSecrets = newSecrets
		}
		ret.CredentialMapping[image] = selectedCreds
	}

	updated := !reflect.DeepEqual(orig.CredentialMapping, ret.CredentialMapping)
	if updated {
		ret.LastUpdatedTime = metav1.Time{Time: time.Now()}
	}
	return *ret, updated
}

func cacheFilePath(dir, image string) string {
	return path.Join(dir, fmt.Sprintf("sha256-%x", sha256.Sum256([]byte(image))))
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
