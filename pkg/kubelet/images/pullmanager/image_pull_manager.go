/*
Copyright 2025 The Kubernetes Authors.

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

package pullmanager

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

var _ ImagePullManager = &PullManager{}

// writeRecordWhileMatchingLimit is a limit at which we stop writing yet-uncached
// records that we found when we were checking if an image pull must be attempted.
// This is to prevent unbounded writes in cases of high namespace turnover.
const writeRecordWhileMatchingLimit = 100

// PullManager is an implementation of the ImagePullManager. It
// tracks images pulled by the kubelet by creating records about ongoing and
// successful pulls.
// It tracks the credentials used with each successful pull in order to be able
// to distinguish tenants requesting access to an image that exists on the kubelet's
// node.
type PullManager struct {
	recordsAccessor PullRecordsAccessor

	imagePolicyEnforcer ImagePullPolicyEnforcer

	imageService kubecontainer.ImageService

	intentAccessors *StripedLockSet // image -> sync.Mutex
	intentCounters  *sync.Map       // image -> number of current in-flight pulls

	pulledAccessors *StripedLockSet // imageRef -> sync.Mutex
}

func NewImagePullManager(ctx context.Context, recordsAccessor PullRecordsAccessor, imagePullPolicy ImagePullPolicyEnforcer, imageService kubecontainer.ImageService, lockStripesNum int32) (*PullManager, error) {
	m := &PullManager{
		recordsAccessor: recordsAccessor,

		imagePolicyEnforcer: imagePullPolicy,

		imageService: imageService,

		intentAccessors: NewStripedLockSet(lockStripesNum),
		intentCounters:  &sync.Map{},

		pulledAccessors: NewStripedLockSet(lockStripesNum),
	}

	m.initialize(ctx)

	return m, nil
}

func (f *PullManager) RecordPullIntent(logger klog.Logger, image string) error {
	f.intentAccessors.Lock(image)
	defer f.intentAccessors.Unlock(image)

	if err := f.recordsAccessor.WriteImagePullIntent(logger, image); err != nil {
		return fmt.Errorf("failed to record image pull intent: %w", err)
	}

	f.incrementIntentCounterForImage(image)
	return nil
}

func (f *PullManager) RecordImagePulled(ctx context.Context, image, imageRef string, credentials *kubeletconfiginternal.ImagePullCredentials) {
	logger := klog.FromContext(ctx)
	if err := f.writePulledRecordIfChanged(ctx, image, imageRef, credentials); err != nil {
		logger.Error(err, "failed to write image pulled record", "imageRef", imageRef)
		return
	}

	// Notice we don't decrement in case of record write error, which leaves dangling
	// imagePullIntents and refCount in the intentCounters map.
	// This is done so that the successfully pulled image is still considered as pulled by the kubelet.
	// The kubelet will attempt to turn the imagePullIntent into a pulled record again when
	// it's restarted.
	f.decrementImagePullIntent(ctx, image)
}

// writePulledRecordIfChanged writes an ImagePulledRecord into the f.pulledDir directory.
// `image` is an image from a container of a Pod object.
// `imageRef` is a reference to the `image“ as used by the CRI.
// `credentials` is a set of credentials that should be written to a new/merged into
// an existing record.
//
// If `credentials` is nil, it marks a situation where an image was pulled under
// unknown circumstances. We should record the image as tracked but no credentials
// should be written in order to force credential verification when the image is
// accessed the next time.
func (f *PullManager) writePulledRecordIfChanged(ctx context.Context, image, imageRef string, credentials *kubeletconfiginternal.ImagePullCredentials) error {
	f.pulledAccessors.Lock(imageRef)
	defer f.pulledAccessors.Unlock(imageRef)
	return f.lockedWritePulledRecordIfChanged(ctx, image, imageRef, credentials)
}

func (f *PullManager) lockedWritePulledRecordIfChanged(ctx context.Context, image, imageRef string, credentials *kubeletconfiginternal.ImagePullCredentials) error {
	logger := klog.FromContext(ctx)

	if credentials.Preloaded && (credentials.NodePodsAccessible ||
		len(credentials.KubernetesSecrets) > 0 ||
		len(credentials.KubernetesServiceAccounts) > 0) {
		panic(fmt.Errorf("tried to write pulled record for image %q that sets both preloaded and credentials: %v", image, credentials))
	}

	sanitizedImage, err := trimImageTagDigest(image)
	if err != nil {
		return fmt.Errorf("invalid image name %q: %w", image, err)
	}

	pulledRecord, _, err := f.recordsAccessor.GetImagePulledRecord(imageRef)
	if err != nil {
		logger.Info("failed to retrieve an ImagePulledRecord", "image", image, "err", err)
		pulledRecord = nil
	}

	var pulledRecordChanged bool
	if pulledRecord == nil {
		pulledRecordChanged = true
		pulledRecord = &kubeletconfiginternal.ImagePulledRecord{
			LastUpdatedTime:   metav1.Time{Time: time.Now()},
			ImageRef:          imageRef,
			CredentialMapping: make(map[string]kubeletconfiginternal.ImagePullCredentials),
		}
		// just the existence of the pulled record for a given imageRef is enough
		// for us to consider it kubelet-pulled. The kubelet should fail safe
		// if it does not find a credential record for the specific image, and it
		// must require credential validation
		if credentials != nil {
			pulledRecord.CredentialMapping[sanitizedImage] = *credentials
		}
	} else {
		pulledRecord, pulledRecordChanged = pulledRecordMergeNewCreds(pulledRecord, sanitizedImage, credentials)
	}

	if !pulledRecordChanged {
		return nil
	}

	return f.recordsAccessor.WriteImagePulledRecord(logger, pulledRecord)
}

func (f *PullManager) RecordImagePullFailed(ctx context.Context, image string) {
	f.decrementImagePullIntent(ctx, image)
}

// decrementImagePullIntent decreses the number of how many times image pull
// intent for a given `image` was requested, and removes the ImagePullIntent file
// if the reference counter for the image reaches zero.
func (f *PullManager) decrementImagePullIntent(ctx context.Context, image string) {
	logger := klog.FromContext(ctx)
	f.intentAccessors.Lock(image)
	defer f.intentAccessors.Unlock(image)

	if f.getIntentCounterForImage(image) <= 1 {
		if err := f.recordsAccessor.DeleteImagePullIntent(logger, image); err != nil {
			logger.Error(err, "failed to remove image pull intent", "image", image)
			return
		}
		// only delete the intent counter once the file was deleted to be consistent
		// with the records
		f.intentCounters.Delete(image)
		return
	}

	f.decrementIntentCounterForImage(image)
}

func (f *PullManager) MustAttemptImagePull(ctx context.Context, image, imageRef string, getPodCredentials GetPodCredentials) (bool, error) {
	if len(imageRef) == 0 {
		return true, nil
	}
	logger := klog.FromContext(ctx)

	var resultForMetrics mustAttemptImagePullResult
	defer func() {
		if len(resultForMetrics) == 0 {
			resultForMetrics = checkResultError
		}
		recordMustAttemptImagePullResult(resultForMetrics)
	}()

	var pulledRecord *kubeletconfiginternal.ImagePulledRecord

	// TODO: sync.OnceValues(---v)
	sanitizedImage, imageSanitizationError := trimImageTagDigest(image)

	pulledRecord, isExempted, err := f.isExemptedByPolicy(ctx, logger, image, sanitizedImage, imageSanitizationError, imageRef)
	if err != nil {
		resultForMetrics = checkResultError
		logger.Error(err, "Unable to access cache records about image pulls")
		return true, nil
	} else if isExempted {
		resultForMetrics = checkResultCredentialPolicyAllowed
		return false, nil
	}

	if pulledRecord == nil {
		// we have no proper records of the image being pulled in the past, we can short-circuit here
		resultForMetrics = checkResultMustAuthenticate
		return true, nil
	}

	if imageSanitizationError != nil {
		resultForMetrics = checkResultError
		logger.Error(imageSanitizationError, "failed to parse image name, forcing image credentials reverification", "image", image)
		return true, nil
	}

	cachedCreds, ok := pulledRecord.CredentialMapping[sanitizedImage]
	if !ok {
		resultForMetrics = checkResultMustAuthenticate
		return true, nil
	}

	if cachedCreds.NodePodsAccessible {
		// anyone on this node can access the image
		resultForMetrics = checkResultCredentialRecordFound
		return false, nil
	}

	if len(cachedCreds.KubernetesSecrets) == 0 && len(cachedCreds.KubernetesServiceAccounts) == 0 {
		resultForMetrics = checkResultMustAuthenticate
		return true, nil
	}

	podSecrets, podServiceAccount, err := getPodCredentials()
	if err != nil {
		resultForMetrics = checkResultError
		return true, err
	}

	for _, podSecret := range podSecrets {
		for _, cachedSecret := range cachedCreds.KubernetesSecrets {

			// we need to check hash len in case hashing failed while storing the record in the keyring
			hashesMatch := len(cachedSecret.CredentialHash) > 0 && podSecret.CredentialHash == cachedSecret.CredentialHash
			secretCoordinatesMatch := podSecret.UID == cachedSecret.UID &&
				podSecret.Namespace == cachedSecret.Namespace &&
				podSecret.Name == cachedSecret.Name

			if hashesMatch {
				if !secretCoordinatesMatch && len(cachedCreds.KubernetesSecrets) < writeRecordWhileMatchingLimit {
					// While we're only matching at this point, we want to ensure this secret is considered valid in the future
					// and so we make an additional write to the cache.
					// writePulledRecord() is a noop in case the secret with the updated hash already appears in the cache.
					if err := f.writePulledRecordIfChanged(ctx, image, imageRef, &kubeletconfiginternal.ImagePullCredentials{KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{podSecret}}); err != nil {
						logger.Error(err, "failed to write an image pulled record", "image", image, "imageRef", imageRef)
					}
				}
				resultForMetrics = checkResultCredentialRecordFound
				return false, nil
			}

			if secretCoordinatesMatch {
				if !hashesMatch && len(cachedCreds.KubernetesSecrets) < writeRecordWhileMatchingLimit {
					// While we're only matching at this point, we want to ensure the updated credentials are considered valid in the future
					// and so we make an additional write to the cache.
					// writePulledRecord() is a noop in case the hash got updated in the meantime.
					if err := f.writePulledRecordIfChanged(ctx, image, imageRef, &kubeletconfiginternal.ImagePullCredentials{KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{podSecret}}); err != nil {
						logger.Error(err, "failed to write an image pulled record", "image", image, "imageRef", imageRef)
					}
					resultForMetrics = checkResultCredentialRecordFound
					return false, nil
				}
			}
		}
	}

	if podServiceAccount != nil && slices.Contains(cachedCreds.KubernetesServiceAccounts, *podServiceAccount) {
		// we found a matching service account, no need to pull the image
		resultForMetrics = checkResultCredentialRecordFound
		return false, nil
	}

	resultForMetrics = checkResultMustAuthenticate
	return true, nil
}

// NOTE: isExemptedByPolicy contains the previous check that was determining whether an
// image was pulled by the kubelet.
// The previous `isPulledByKubelet` code was already locking all the records we
// need to determine whether we should write a `preloaded:true` pulled record
func (f *PullManager) isExemptedByPolicy(ctx context.Context, logger klog.Logger, image, sanitizedImage string, imageSanitizationError error, imageRef string) (*kubeletconfiginternal.ImagePulledRecord, bool, error) {
	// don't allow changes to the files we're using for our decision
	f.pulledAccessors.Lock(imageRef)
	defer f.pulledAccessors.Unlock(imageRef)
	f.intentAccessors.Lock(image)
	defer f.intentAccessors.Unlock(image)

	var pulledRecord *kubeletconfiginternal.ImagePulledRecord
	var intentExists *bool
	isPulledByKubelet, err := func() (bool, error) {
		var exists bool
		var err error
		pulledRecord, exists, err = f.recordsAccessor.GetImagePulledRecord(imageRef)
		if err != nil {
			return false, err
		}

		if exists && pulledRecord == nil {
			return true, nil
		}

		if pulledRecord != nil {
			if imageSanitizationError != nil {
				// we're not able to check the credentials of the ImagePulledRecord
				return true, nil
			}

			credential, ok := pulledRecord.CredentialMapping[sanitizedImage]
			if ok && !credential.Preloaded {
				return true, nil
			}
			if !ok {
				// check if there are any !Preloaded records -> mark as pulled by
				// kubelet if at least one exists
				// NOTE: this would be easier if pulled records were separate
				for _, c := range pulledRecord.CredentialMapping {
					if !c.Preloaded {
						return true, nil
					}
				}
			}

			intentExistsVal, err := f.lockedImagePullIntentExists(image)
			if err != nil {
				return false, err
			}
			intentExists = new(intentExistsVal)

			if intentExistsVal {
				return true, nil
			}

			// pulled record says preloaded, we don't have any record of the image being pulled right now
			return false, nil
		}

		// pulled record does not exist
		intentExistsVal, err := f.lockedImagePullIntentExists(image)
		if err != nil {
			return false, err
		}
		intentExists = new(intentExistsVal)
		return intentExistsVal, nil
	}()
	if err != nil {
		return nil, false, err
	}

	if !f.imagePolicyEnforcer.RequireCredentialVerificationForImage(image, isPulledByKubelet) {
		// TODO: create a record that this was exempt/pre-pulled - record "NotPulledByKubelet"
		// what happens when there suddenly is a pulled record, but not just for the imageName?
		//--------------------------------------------------
		// NOTE: needs to decide if all conditions are satisfied for the write - no pull intents, no `preloaded:false` records
		//		  Concurrency - what happens if these get written in the meantime? We cannot lock because the write itself locks - should
		//						this check be part of the ImageCreds merging?
		//						The write locks pulled records, but we also need to check intents...
		//						|--> we probably need a double-locking method
		if f.lockedShouldWriteImagePreloaded(intentExists, pulledRecord, image) {
			if err := f.lockedWritePulledRecordIfChanged(ctx, image, imageRef, &kubeletconfiginternal.ImagePullCredentials{Preloaded: true}); err != nil {
				logger.Error(err, "failed to record image as preloaded", "image", image, "imageRef", imageRef)
			}
		}
		return pulledRecord, true, nil
	}

	return pulledRecord, false, nil
}

// lockedImagePullIntentExists is an optimized check for pull intent existence.
// It checks the intent number, however, if it's zero it may only mean kubelet
// restarted since writing the intent record and we must fall back to the
// actual cache
func (f *PullManager) lockedImagePullIntentExists(image string) (bool, error) {
	if f.getIntentCounterForImage(image) > 0 {
		return true, nil
	}

	exists, err := f.recordsAccessor.ImagePullIntentExists(image)
	if err != nil {
		return false, fmt.Errorf("failed to check existence of an image pull intent: %w", err)
	}
	return exists, nil
}

func (f *PullManager) PruneUnknownRecords(ctx context.Context, imageList []string, until time.Time) {
	f.pulledAccessors.GlobalLock()
	defer f.pulledAccessors.GlobalUnlock()

	logger := klog.FromContext(ctx)
	pulledRecords, err := f.recordsAccessor.ListImagePulledRecords()
	if err != nil {
		logger.Error(err, "there were errors listing ImagePulledRecords, garbage collection will proceed with incomplete records list")
	}

	imagesInUse := sets.New(imageList...)
	for _, imageRecord := range pulledRecords {
		if !imageRecord.LastUpdatedTime.Time.Before(until) {
			// the image record was only updated after the GC started
			continue
		}

		if imagesInUse.Has(imageRecord.ImageRef) {
			continue
		}

		if err := f.recordsAccessor.DeleteImagePulledRecord(logger, imageRecord.ImageRef); err != nil {
			logger.Error(err, "failed to remove an ImagePulledRecord", "imageRef", imageRecord.ImageRef)
		}
	}

}

// initialize gathers all the images from pull intent records that exist
// from the previous kubelet runs.
// If the CRI reports any of the above images as already pulled, we turn the
// pull intent into a pulled record and the original pull intent is deleted.
//
// This method is not thread-safe and it should only be called upon the creation
// of the PullManager.
func (f *PullManager) initialize(ctx context.Context) {
	logger := klog.FromContext(ctx)
	pullIntents, err := f.recordsAccessor.ListImagePullIntents()
	if err != nil {
		logger.Error(err, "there were errors listing ImagePullIntents, continuing with an incomplete records list")
	}

	if len(pullIntents) == 0 {
		return
	}

	imageObjs, err := f.imageService.ListImages(ctx)
	if err != nil {
		logger.Error(err, "failed to list images")
	}

	inFlightPulls := sets.New[string]()
	for _, intent := range pullIntents {
		inFlightPulls.Insert(intent.Image)
	}

	// Each of the images known to the CRI might consist of multiple tags and digests,
	// which is what we track in the ImagePullIntent - we need to go through all of these
	// for each image.
	for _, imageObj := range imageObjs {
		existingRecordedImages := searchForExistingTagDigest(inFlightPulls, imageObj)

		for _, image := range existingRecordedImages.UnsortedList() {

			if err := f.writePulledRecordIfChanged(ctx, image, imageObj.ID, nil); err != nil {
				logger.Error(err, "failed to write an image pull record", "imageRef", imageObj.ID)
				continue
			}

			if err := f.recordsAccessor.DeleteImagePullIntent(logger, image); err != nil {
				logger.V(2).Info("failed to remove image pull intent file", "imageName", image, "error", err)
			}
		}
	}

}

func (f *PullManager) incrementIntentCounterForImage(image string) {
	f.intentCounters.Store(image, f.getIntentCounterForImage(image)+1)
}
func (f *PullManager) decrementIntentCounterForImage(image string) {
	f.intentCounters.Store(image, f.getIntentCounterForImage(image)-1)
}

func (f *PullManager) getIntentCounterForImage(image string) int32 {
	intentNumAny, ok := f.intentCounters.Load(image)
	if !ok {
		return 0
	}
	intentNum, ok := intentNumAny.(int32)
	if !ok {
		panic(fmt.Sprintf("expected the intentCounters sync map to only contain int32 values, got %T", intentNumAny))
	}
	return intentNum
}

// searchForExistingTagDigest loops through the `image` RepoDigests and RepoTags
// and tries to find all image digests/tags in `inFlightPulls`, which is a map of
// containerImage -> pulling intent path.
func searchForExistingTagDigest(inFlightPulls sets.Set[string], image kubecontainer.Image) sets.Set[string] {
	existingRecordedImages := sets.New[string]()
	for _, digest := range image.RepoDigests {
		if ok := inFlightPulls.Has(digest); ok {
			existingRecordedImages.Insert(digest)
		}
	}

	for _, tag := range image.RepoTags {
		if ok := inFlightPulls.Has(tag); ok {
			existingRecordedImages.Insert(tag)
		}
	}

	return existingRecordedImages
}

func (f *PullManager) lockedShouldWriteImagePreloaded(intentExists *bool, pulledRecord *kubeletconfiginternal.ImagePulledRecord, image string) bool {
	if pulledRecord != nil {
		for _, creds := range pulledRecord.CredentialMapping {
			if !creds.Preloaded {
				return false
			}
		}
		return true
	}

	if intentExists == nil {
		intentExistsVal, err := f.lockedImagePullIntentExists(image)
		if err != nil {
			// TODO: log here
			return false
		}
		intentExists = new(intentExistsVal)
	}

	return !*intentExists
}

type kubeSecretCoordinates struct {
	UID       string
	Namespace string
	Name      string
}

// pulledRecordMergeNewCreds merges the credentials from `newCreds` into the `orig`
// record for the `imageNoTagDigest` image.
// `imageNoTagDigest` is the content of the `image` field from a pod's container
// after any tag or digest were removed from it.
//
// NOTE: pulledRecordMergeNewCreds() may be often called in the read path of
// PullManager.MustAttemptImagePull() and so it's desirable to limit allocations
// (e.g. DeepCopy()) until it is necessary.
func pulledRecordMergeNewCreds(orig *kubeletconfiginternal.ImagePulledRecord, imageNoTagDigest string, newCreds *kubeletconfiginternal.ImagePullCredentials) (*kubeletconfiginternal.ImagePulledRecord, bool) {
	if newCreds == nil {
		// no new credential information to record
		return orig, false
	}

	if !newCreds.Preloaded && !newCreds.NodePodsAccessible && len(newCreds.KubernetesSecrets) == 0 && len(newCreds.KubernetesServiceAccounts) == 0 {
		// we don't have any secret, service account credentials or node-wide access to record
		return orig, false
	}
	selectedCreds, found := orig.CredentialMapping[imageNoTagDigest]
	if !found {
		ret := orig.DeepCopy()
		if ret.CredentialMapping == nil {
			ret.CredentialMapping = make(map[string]kubeletconfiginternal.ImagePullCredentials)
		}
		ret.CredentialMapping[imageNoTagDigest] = *newCreds
		ret.LastUpdatedTime = metav1.Time{Time: time.Now()}
		return ret, true
	}

	if selectedCreds.NodePodsAccessible {
		return orig, false
	}

	// preloaded can only ever be true if this was a new record or if both the
	// previous and current state set preloaded=true
	if newCreds.Preloaded && selectedCreds.Preloaded {
		return orig, false
	}
	selectedCreds.Preloaded = false

	switch {
	case newCreds.NodePodsAccessible:
		selectedCreds.NodePodsAccessible = true
		selectedCreds.KubernetesSecrets = nil
		selectedCreds.KubernetesServiceAccounts = nil

		ret := orig.DeepCopy()
		ret.CredentialMapping[imageNoTagDigest] = selectedCreds
		ret.LastUpdatedTime = metav1.Time{Time: time.Now()}
		return ret, true

	case len(newCreds.KubernetesSecrets) > 0:
		var secretsChanged bool
		selectedCreds.KubernetesSecrets, secretsChanged = mergePullSecrets(selectedCreds.KubernetesSecrets, newCreds.KubernetesSecrets)
		if !secretsChanged {
			return orig, false
		}

	case len(newCreds.KubernetesServiceAccounts) > 0:
		var serviceAccountsChanged bool
		selectedCreds.KubernetesServiceAccounts, serviceAccountsChanged = mergePullServiceAccounts(selectedCreds.KubernetesServiceAccounts, newCreds.KubernetesServiceAccounts)
		if !serviceAccountsChanged {
			return orig, false
		}
	}

	ret := orig.DeepCopy()
	ret.CredentialMapping[imageNoTagDigest] = selectedCreds
	ret.LastUpdatedTime = metav1.Time{Time: time.Now()}
	return ret, true
}

// mergePullSecrets merges two slices of ImagePullSecret object into one while
// keeping the objects unique per `Namespace, Name, UID` key.
//
// In case an object from the `new` slice has the same `Namespace, Name, UID` combination
// as an object from `orig`, the result will use the CredentialHash value of the
// object from `new`.
//
// The returned slice is sorted by Namespace, Name and UID (in this order). Also
// returns an indicator whether the set of input secrets chaged.
func mergePullSecrets(orig, new []kubeletconfiginternal.ImagePullSecret) ([]kubeletconfiginternal.ImagePullSecret, bool) {
	credSet := make(map[kubeSecretCoordinates]string)
	for _, secret := range orig {
		credSet[kubeSecretCoordinates{
			UID:       secret.UID,
			Namespace: secret.Namespace,
			Name:      secret.Name,
		}] = secret.CredentialHash
	}

	changed := false
	for _, s := range new {
		key := kubeSecretCoordinates{UID: s.UID, Namespace: s.Namespace, Name: s.Name}
		if existingHash, ok := credSet[key]; !ok || existingHash != s.CredentialHash {
			changed = true
			credSet[key] = s.CredentialHash
		}
	}
	if !changed {
		return orig, false
	}

	ret := make([]kubeletconfiginternal.ImagePullSecret, 0, len(credSet))
	for coords, hash := range credSet {
		ret = append(ret, kubeletconfiginternal.ImagePullSecret{
			UID:            coords.UID,
			Namespace:      coords.Namespace,
			Name:           coords.Name,
			CredentialHash: hash,
		})
	}
	// we don't need to use the stable version because secret coordinates used for ordering are unique in the set
	slices.SortFunc(ret, imagePullSecretLess)

	return ret, true
}

// imagePullSecretLess is a helper function to define ordering in a slice of
// ImagePullSecret objects.
func imagePullSecretLess(a, b kubeletconfiginternal.ImagePullSecret) int {
	if cmp := strings.Compare(a.Namespace, b.Namespace); cmp != 0 {
		return cmp
	}

	if cmp := strings.Compare(a.Name, b.Name); cmp != 0 {
		return cmp
	}

	return strings.Compare(a.UID, b.UID)
}

// mergePullServiceAccounts merges two slices of ImagePullServiceAccount object into one while
// keeping the objects unique per `Namespace, Name, UID` key.
// The returned slice is sorted by Namespace, Name and UID (in this order).
// Also returns an indicator whether the set of input service accounts changed.
func mergePullServiceAccounts(orig, new []kubeletconfiginternal.ImagePullServiceAccount) ([]kubeletconfiginternal.ImagePullServiceAccount, bool) {
	credSet := sets.New[kubeletconfiginternal.ImagePullServiceAccount]()
	for _, serviceAccount := range orig {
		credSet.Insert(serviceAccount)
	}

	changed := false
	for _, s := range new {
		if !credSet.Has(s) {
			changed = true
			credSet.Insert(s)
		}
	}
	if !changed {
		return orig, false
	}

	ret := credSet.UnsortedList()
	slices.SortFunc(ret, imagePullServiceAccountLess)

	return ret, true
}

// imagePullServiceAccountLess is a helper function to define ordering in a slice of
// ImagePullServiceAccount objects.
func imagePullServiceAccountLess(a, b kubeletconfiginternal.ImagePullServiceAccount) int {
	if cmp := strings.Compare(a.Namespace, b.Namespace); cmp != 0 {
		return cmp
	}

	if cmp := strings.Compare(a.Name, b.Name); cmp != 0 {
		return cmp
	}

	if cmp := strings.Compare(a.UID, b.UID); cmp != 0 {
		return cmp
	}

	return 0
}
