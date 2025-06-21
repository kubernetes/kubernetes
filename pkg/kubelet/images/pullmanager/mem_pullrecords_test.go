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
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

type testPullRecordsAccessor struct {
	intents         map[string]*kubeletconfiginternal.ImagePullIntent
	intentListError error

	pulledRecords          map[string]*kubeletconfiginternal.ImagePulledRecord
	pulledRecordsListError error

	nextGetWriteError error
}

func (a *testPullRecordsAccessor) ListImagePullIntents() ([]*kubeletconfiginternal.ImagePullIntent, error) {
	ret := make([]*kubeletconfiginternal.ImagePullIntent, 0, len(a.intents))
	for _, intent := range a.intents {
		ret = append(ret, intent)
	}

	slices.SortFunc(ret, pullIntentsCmp)
	return ret, a.intentListError
}

func (a *testPullRecordsAccessor) ListImagePulledRecords() ([]*kubeletconfiginternal.ImagePulledRecord, error) {
	ret := make([]*kubeletconfiginternal.ImagePulledRecord, 0, len(a.pulledRecords))
	for _, record := range a.pulledRecords {
		ret = append(ret, record)
	}

	slices.SortFunc(ret, pulledRecordsCmp)
	return ret, a.pulledRecordsListError
}

func (a *testPullRecordsAccessor) ImagePullIntentExists(image string) (bool, error) {
	if a.nextGetWriteError != nil {
		return false, a.nextGetWriteError
	}
	_, exists := a.intents[image]

	return exists, nil
}

func (a *testPullRecordsAccessor) GetImagePulledRecord(imageRef string) (*kubeletconfiginternal.ImagePulledRecord, bool, error) {
	record, exists := a.pulledRecords[imageRef]
	if a.nextGetWriteError != nil {
		return nil, exists, a.nextGetWriteError
	}
	return record, exists, nil
}

func (a *testPullRecordsAccessor) WriteImagePullIntent(image string) error {
	if a.nextGetWriteError != nil {
		return a.nextGetWriteError
	}

	a.intents[image] = &kubeletconfiginternal.ImagePullIntent{
		Image: image,
	}
	return nil
}
func (a *testPullRecordsAccessor) WriteImagePulledRecord(rec *kubeletconfiginternal.ImagePulledRecord) error {
	if a.nextGetWriteError != nil {
		return a.nextGetWriteError
	}

	a.pulledRecords[rec.ImageRef] = rec
	return nil
}

func (a *testPullRecordsAccessor) DeleteImagePullIntent(image string) error {
	if a.nextGetWriteError != nil {
		return a.nextGetWriteError
	}

	delete(a.intents, image)
	return nil
}
func (a *testPullRecordsAccessor) DeleteImagePulledRecord(imageRef string) error {
	if a.nextGetWriteError != nil {
		return a.nextGetWriteError
	}

	delete(a.pulledRecords, imageRef)
	return nil
}

func (a *testPullRecordsAccessor) withIntentsListError(err error) *testPullRecordsAccessor {
	a.intentListError = err
	return a
}

func (a *testPullRecordsAccessor) withPulledRecordsListError(err error) *testPullRecordsAccessor {
	a.pulledRecordsListError = err
	return a
}

func (a *testPullRecordsAccessor) withNextGetWriteError(err error) *testPullRecordsAccessor {
	a.nextGetWriteError = err
	return a
}

func newTestPullRecordsAccessor(
	pullIntents []*kubeletconfiginternal.ImagePullIntent,
	pulledRecords []*kubeletconfiginternal.ImagePulledRecord,
) *testPullRecordsAccessor {
	cachedPullIntents := make(map[string]*kubeletconfiginternal.ImagePullIntent)
	for _, intent := range pullIntents {
		cachedPullIntents[intent.Image] = intent
	}

	cachedPullRecords := make(map[string]*kubeletconfiginternal.ImagePulledRecord)
	for _, record := range pulledRecords {
		cachedPullRecords[record.ImageRef] = record
	}

	return &testPullRecordsAccessor{
		intents:       cachedPullIntents,
		pulledRecords: cachedPullRecords,
	}
}

type recordingPullRecordsAccessor struct {
	delegate      PullRecordsAccessor
	methodsCalled []string
}

func (a *recordingPullRecordsAccessor) ListImagePullIntents() ([]*kubeletconfiginternal.ImagePullIntent, error) {
	a.methodsCalled = append(a.methodsCalled, "ListImagePullIntents")
	return a.delegate.ListImagePullIntents()
}

func (a *recordingPullRecordsAccessor) ListImagePulledRecords() ([]*kubeletconfiginternal.ImagePulledRecord, error) {
	a.methodsCalled = append(a.methodsCalled, "ListImagePulledRecords")
	return a.delegate.ListImagePulledRecords()
}

func (a *recordingPullRecordsAccessor) ImagePullIntentExists(image string) (bool, error) {
	a.methodsCalled = append(a.methodsCalled, "ImagePullIntentExists")
	return a.delegate.ImagePullIntentExists(image)
}

func (a *recordingPullRecordsAccessor) GetImagePulledRecord(imageRef string) (*kubeletconfiginternal.ImagePulledRecord, bool, error) {
	a.methodsCalled = append(a.methodsCalled, "GetImagePulledRecord")
	return a.delegate.GetImagePulledRecord(imageRef)
}

func (a *recordingPullRecordsAccessor) WriteImagePullIntent(image string) error {
	a.methodsCalled = append(a.methodsCalled, "WriteImagePullIntent")
	return a.delegate.WriteImagePullIntent(image)
}

func (a *recordingPullRecordsAccessor) WriteImagePulledRecord(rec *kubeletconfiginternal.ImagePulledRecord) error {
	a.methodsCalled = append(a.methodsCalled, "WriteImagePulledRecord")
	return a.delegate.WriteImagePulledRecord(rec)
}

func (a *recordingPullRecordsAccessor) DeleteImagePullIntent(image string) error {
	a.methodsCalled = append(a.methodsCalled, "DeleteImagePullIntent")
	return a.delegate.DeleteImagePullIntent(image)
}

func (a *recordingPullRecordsAccessor) DeleteImagePulledRecord(imageRef string) error {
	a.methodsCalled = append(a.methodsCalled, "DeleteImagePulledRecord")
	return a.delegate.DeleteImagePulledRecord(imageRef)
}

func TestNewCachedPullRecordsAccessor(t *testing.T) {
	tests := []struct {
		name                   string
		delegate               *testPullRecordsAccessor
		wantIntents            []*kubeletconfiginternal.ImagePullIntent
		wantIntentsError       error
		wantPulledRecords      []*kubeletconfiginternal.ImagePulledRecord
		wantPulledRecordsError error
	}{
		{
			name: "no issues during init",
			delegate: newTestPullRecordsAccessor(
				testPullIntents(),
				testPulledRecords(),
			),
			wantIntents:       testPullIntentsSorted(),
			wantPulledRecords: testPulledRecordsSorted(),
		},
		{
			name: "delegate intents list error",
			delegate: newTestPullRecordsAccessor(
				testPullIntents(),
				testPulledRecords(),
			).withIntentsListError(fmt.Errorf("intentsError")),
			wantIntents:       testPullIntentsSorted(),
			wantIntentsError:  fmt.Errorf("intentsError"),
			wantPulledRecords: testPulledRecordsSorted(),
		},
		{
			name: "delegate pulledRecords list error",
			delegate: newTestPullRecordsAccessor(
				testPullIntents(),
				testPulledRecords(),
			).withPulledRecordsListError(fmt.Errorf("pulledRecordsError")),
			wantIntents:            testPullIntentsSorted(),
			wantPulledRecords:      testPulledRecordsSorted(),
			wantPulledRecordsError: fmt.Errorf("pulledRecordsError"),
		},
		{
			name: "both delegate lists fail during init",
			delegate: newTestPullRecordsAccessor(
				testPullIntents(),
				testPulledRecords(),
			).withPulledRecordsListError(fmt.Errorf("pulledRecordsError")).
				withIntentsListError(fmt.Errorf("intentsError")),
			wantIntents:            testPullIntentsSorted(),
			wantIntentsError:       fmt.Errorf("intentsError"),
			wantPulledRecords:      testPulledRecordsSorted(),
			wantPulledRecordsError: fmt.Errorf("pulledRecordsError"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotAccessor := NewCachedPullRecordsAccessor(tt.delegate)
			if expectedIntents := tt.delegate.intents; !reflect.DeepEqual(gotAccessor.intents, expectedIntents) {
				t.Errorf("NewCachedPullRecordsAccessor intent records = %v, want %v", gotAccessor.intents, expectedIntents)
			}

			gotIntents, intentsErr := gotAccessor.ListImagePullIntents()
			if !reflect.DeepEqual(gotIntents, tt.wantIntents) {
				t.Errorf("NewCachedPullRecordsAccessor().ListImagePullIntents() = %v, want %v", gotIntents, tt.wantIntents)
			}

			if !cmpErrorStrings(intentsErr, tt.wantIntentsError) {
				t.Errorf("NewCachedPullRecordsAccessor().ListImagePullIntents() errors don't match = %v, want %v", intentsErr, tt.wantIntentsError)
			}

			if expectedPulledRecords := tt.delegate.pulledRecords; !reflect.DeepEqual(gotAccessor.pulledRecords, expectedPulledRecords) {
				t.Errorf("NewCachedPullRecordsAccessor pulled records = %v, want %v", gotAccessor.pulledRecords, expectedPulledRecords)
			}

			gotPulledRecords, pulledRecordsErr := gotAccessor.ListImagePulledRecords()
			if !reflect.DeepEqual(gotPulledRecords, tt.wantPulledRecords) {
				t.Errorf("NewCachedPullRecordsAccessor().ListImagePulledRecords() = %v, want %v", gotPulledRecords, tt.wantPulledRecords)
			}

			if !cmpErrorStrings(pulledRecordsErr, tt.wantPulledRecordsError) {
				t.Errorf("NewCachedPullRecordsAccessor().ListImagePullIntents() errors don't match = %v, want %v", pulledRecordsErr, tt.wantPulledRecordsError)
			}
		})
	}
}

func Test_cachedPullRecordsAccessor_ListImagePullIntents(t *testing.T) {
	tests := []struct {
		name                 string
		delegate             *testPullRecordsAccessor
		initialCache         map[string]*kubeletconfiginternal.ImagePullIntent
		initialError         error
		wantListIntents      []*kubeletconfiginternal.ImagePullIntent
		wantListIntentsError error
		wantCacheIntents     map[string]*kubeletconfiginternal.ImagePullIntent
	}{
		{
			name: "no error, intents listed from cache",
			delegate: newTestPullRecordsAccessor(
				[]*kubeletconfiginternal.ImagePullIntent{
					{
						Image: "this would be wrong",
					},
				},
				nil,
			).withIntentsListError(fmt.Errorf("should not have occurred")),
			initialCache:     pullIntentsToMap(testPullIntents()),
			wantListIntents:  testPullIntentsSorted(),
			wantCacheIntents: pullIntentsToMap(testPullIntents()),
		},
		{
			name: "initial cache error gets cleared by successful intents list from delegate",
			delegate: newTestPullRecordsAccessor(
				testPullIntents(),
				nil,
			),
			initialCache: pullIntentsToMap([]*kubeletconfiginternal.ImagePullIntent{
				{Image: "this would be wrong"},
			}),
			initialError:     fmt.Errorf("initial intents error"),
			wantListIntents:  testPullIntentsSorted(),
			wantCacheIntents: pullIntentsToMap(testPullIntents()),
		},
		{
			name: "initial cache error, partial intents listed from delegate with an error",
			delegate: newTestPullRecordsAccessor(
				testPullIntents(),
				nil,
			).withIntentsListError(fmt.Errorf("only partial list was returned")),
			initialCache: pullIntentsToMap([]*kubeletconfiginternal.ImagePullIntent{
				{Image: "this would be wrong"},
			}),
			initialError:         fmt.Errorf("initial intents error"),
			wantListIntents:      testPullIntentsSorted(),
			wantCacheIntents:     pullIntentsToMap(testPullIntents()),
			wantListIntentsError: fmt.Errorf("only partial list was returned"),
		},
		{
			name:             "delegate returns empty list",
			delegate:         newTestPullRecordsAccessor(nil, nil),
			initialCache:     pullIntentsToMap(testPullIntents()),
			initialError:     fmt.Errorf("initial intents error"),
			wantListIntents:  []*kubeletconfiginternal.ImagePullIntent{},
			wantCacheIntents: map[string]*kubeletconfiginternal.ImagePullIntent{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cachedAccessor := &cachedPullRecordsAccessor{
				delegate:        tt.delegate,
				intentsMutex:    &sync.RWMutex{},
				intents:         tt.initialCache,
				intentListError: tt.initialError,
			}

			gotIntents, gotErr := cachedAccessor.ListImagePullIntents()
			if !reflect.DeepEqual(gotIntents, tt.wantListIntents) {
				t.Errorf("ListImagePullIntents() = %v, want %v", gotIntents, tt.wantListIntents)
			}

			if !cmpErrorStrings(gotErr, tt.wantListIntentsError) {
				t.Errorf("ListImagePullIntents() error = %v, want %v", gotErr, tt.wantListIntentsError)
			}

			if !reflect.DeepEqual(cachedAccessor.intents, tt.wantCacheIntents) {
				t.Errorf("ListImagePullIntents() cache state = %v, want %v", cachedAccessor.intents, tt.wantCacheIntents)
			}

			if !cmpErrorStrings(cachedAccessor.intentListError, tt.wantListIntentsError) {
				t.Errorf("ListImagePullIntents() cached error = %v, want %v", cachedAccessor.intentListError, tt.wantListIntentsError)
			}
		})
	}
}

func Test_cachedPullRecordsAccessor_ListImagePulledRecords(t *testing.T) {
	tests := []struct {
		name                       string
		delegate                   *testPullRecordsAccessor
		initialCache               map[string]*kubeletconfiginternal.ImagePulledRecord
		initialError               error
		wantListPulledRecords      []*kubeletconfiginternal.ImagePulledRecord
		wantListPulledRecordsError error
		wantCachePulledRecords     map[string]*kubeletconfiginternal.ImagePulledRecord
	}{
		{
			name: "no error, imagePulledRecords listed from cache",
			delegate: newTestPullRecordsAccessor(
				nil,
				[]*kubeletconfiginternal.ImagePulledRecord{
					{ImageRef: "this would be wrong"},
				},
			).withPulledRecordsListError(fmt.Errorf("should not have occurred")),
			initialCache:           pulledRecordsToMap(testPulledRecords()),
			wantListPulledRecords:  testPulledRecordsSorted(),
			wantCachePulledRecords: pulledRecordsToMap(testPulledRecords()),
		},
		{
			name: "initial cache error gets cleared by successful imagePulledRecords list from delegate",
			delegate: newTestPullRecordsAccessor(
				nil,
				testPulledRecords(),
			),
			initialCache: pulledRecordsToMap([]*kubeletconfiginternal.ImagePulledRecord{
				{ImageRef: "this would be wrong"},
			}),
			initialError:           fmt.Errorf("initial imagePulledRecords error"),
			wantListPulledRecords:  testPulledRecordsSorted(),
			wantCachePulledRecords: pulledRecordsToMap(testPulledRecords()),
		},
		{
			name: "initial cache error, partial imagePulledRecords listed from delegate with an error",
			delegate: newTestPullRecordsAccessor(
				nil,
				testPulledRecords(),
			).withPulledRecordsListError(fmt.Errorf("only partial list was returned")),
			initialCache: pulledRecordsToMap([]*kubeletconfiginternal.ImagePulledRecord{
				{ImageRef: "this would be wrong"},
			}),
			initialError:               fmt.Errorf("initial imagePulledRecords error"),
			wantListPulledRecords:      testPulledRecordsSorted(),
			wantCachePulledRecords:     pulledRecordsToMap(testPulledRecords()),
			wantListPulledRecordsError: fmt.Errorf("only partial list was returned"),
		},
		{
			name:                   "delegate returns empty list",
			delegate:               newTestPullRecordsAccessor(nil, nil),
			initialCache:           pulledRecordsToMap(testPulledRecords()),
			initialError:           fmt.Errorf("initial pull records error"),
			wantListPulledRecords:  []*kubeletconfiginternal.ImagePulledRecord{},
			wantCachePulledRecords: map[string]*kubeletconfiginternal.ImagePulledRecord{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cachedAccessor := &cachedPullRecordsAccessor{
				delegate:               tt.delegate,
				pulledRecordsMutex:     &sync.RWMutex{},
				pulledRecords:          tt.initialCache,
				pulledRecordsListError: tt.initialError,
			}

			gotPulledRecords, gotErr := cachedAccessor.ListImagePulledRecords()
			if !reflect.DeepEqual(gotPulledRecords, tt.wantListPulledRecords) {
				t.Errorf("ListImagePulledRecords() = %v, want %v", gotPulledRecords, tt.wantListPulledRecords)
			}

			if !cmpErrorStrings(gotErr, tt.wantListPulledRecordsError) {
				t.Errorf("ListImagePulledRecords() error = %v, want %v", gotErr, tt.wantListPulledRecordsError)
			}

			if !reflect.DeepEqual(cachedAccessor.pulledRecords, tt.wantCachePulledRecords) {
				t.Errorf("ListImagePulledRecords() cache state = %v, want %v", cachedAccessor.pulledRecords, tt.wantCachePulledRecords)
			}

			if !cmpErrorStrings(cachedAccessor.pulledRecordsListError, tt.wantListPulledRecordsError) {
				t.Errorf("ListImagePulledRecords() cached error = %v, want %v", cachedAccessor.pulledRecordsListError, tt.wantListPulledRecordsError)
			}
		})
	}
}

func Test_cachedPullRecordsAccessor_ImagePullIntentExists(t *testing.T) {
	tests := []struct {
		name             string
		delegate         *testPullRecordsAccessor
		initialCache     map[string]*kubeletconfiginternal.ImagePullIntent
		inputImage       string
		want             bool
		wantErr          error
		wantCacheIntents map[string]*kubeletconfiginternal.ImagePullIntent
	}{
		{
			name: "intent exists in cache",
			delegate: newTestPullRecordsAccessor(nil, nil).
				withNextGetWriteError(fmt.Errorf("should not have occurred")),
			initialCache:     pullIntentsToMap(testPullIntents()),
			inputImage:       "test.repo/org1/project1:tag",
			want:             true,
			wantCacheIntents: pullIntentsToMap(testPullIntents()),
		},
		{
			name:     "intent exists in delegate, cache gets updated",
			delegate: newTestPullRecordsAccessor(testPullIntents(), nil),
			initialCache: pullIntentsToMap([]*kubeletconfiginternal.ImagePullIntent{
				{Image: "test.repo/org1/project3:tag"},
				{Image: "test.repo/org2/project1:tag"},
			}),
			inputImage: "test.repo/org1/project1:tag",
			want:       true,
			wantCacheIntents: pullIntentsToMap([]*kubeletconfiginternal.ImagePullIntent{
				{Image: "test.repo/org1/project3:tag"},
				{Image: "test.repo/org2/project1:tag"},
				{Image: "test.repo/org1/project1:tag"},
			}),
		},
		{
			name:             "intent does not exist in cache or delegate",
			delegate:         newTestPullRecordsAccessor(testPullIntents(), nil),
			initialCache:     pullIntentsToMap(testPullIntents()),
			inputImage:       "test.repo/org1/doesnotexist:tag",
			want:             false,
			wantCacheIntents: pullIntentsToMap(testPullIntents()),
		},
		{
			name: "intent does not exist in cache, delegate returns error on GET",
			delegate: newTestPullRecordsAccessor(testPullIntents(), nil).
				withNextGetWriteError(fmt.Errorf("record malformed")),
			initialCache:     pullIntentsToMap(testPullIntents()),
			inputImage:       "test.repo/org1/doesnotexist:tag",
			want:             false,
			wantErr:          fmt.Errorf("record malformed"),
			wantCacheIntents: pullIntentsToMap(testPullIntents()),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &cachedPullRecordsAccessor{
				delegate:     tt.delegate,
				intentsMutex: &sync.RWMutex{},
				intents:      tt.initialCache,
			}
			got, err := c.ImagePullIntentExists(tt.inputImage)
			if !cmpErrorStrings(err, tt.wantErr) {
				t.Errorf("cachedPullRecordsAccessor.ImagePullIntentExists() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("cachedPullRecordsAccessor.ImagePullIntentExists() = %v, want %v", got, tt.want)
			}

			if !reflect.DeepEqual(c.intents, tt.wantCacheIntents) {
				t.Errorf("cachedPullRecordsAccessor.ImagePullIntentExists() cache state = %v, want %v", c.intents, tt.wantCacheIntents)
			}
		})
	}
}

func Test_cachedPullRecordsAccessor_WriteImagePullIntent(t *testing.T) {
	tests := []struct {
		name                string
		delegate            *recordingPullRecordsAccessor
		initialCache        map[string]*kubeletconfiginternal.ImagePullIntent
		inputImage          string
		wantErr             error
		wantCacheIntents    map[string]*kubeletconfiginternal.ImagePullIntent
		wantDelegateIntents map[string]*kubeletconfiginternal.ImagePullIntent
		wantRecordedMethods []string
	}{
		{
			name: "successful write through an empty cache to delegate",
			delegate: &recordingPullRecordsAccessor{
				delegate: newTestPullRecordsAccessor(nil, nil),
			},
			initialCache: map[string]*kubeletconfiginternal.ImagePullIntent{},
			inputImage:   "test.repo/org1/project1:tag",
			wantCacheIntents: pullIntentsToMap([]*kubeletconfiginternal.ImagePullIntent{
				{Image: "test.repo/org1/project1:tag"},
			}),
			wantRecordedMethods: []string{"WriteImagePullIntent"},
		},
		{
			name: "rewrite cached intent, write through to delegate",
			delegate: &recordingPullRecordsAccessor{
				delegate: newTestPullRecordsAccessor(nil, nil),
			},
			initialCache: map[string]*kubeletconfiginternal.ImagePullIntent{
				"test.repo/org1/project1:tag": {Image: "something else"}, // this is not a real scenario but we're testing the cache can be overridden
			},
			inputImage: "test.repo/org1/project1:tag",
			wantCacheIntents: pullIntentsToMap([]*kubeletconfiginternal.ImagePullIntent{
				{Image: "test.repo/org1/project1:tag"},
			}),
			wantRecordedMethods: []string{"WriteImagePullIntent"},
		},
		{
			name: "write even though the very same intent is already in the cache",
			delegate: &recordingPullRecordsAccessor{
				delegate: newTestPullRecordsAccessor(nil, nil),
			},
			initialCache: map[string]*kubeletconfiginternal.ImagePullIntent{
				"test.repo/org1/project1:tag": {Image: "test.repo/org1/project1:tag"},
			},
			inputImage: "test.repo/org1/project1:tag",
			wantCacheIntents: pullIntentsToMap([]*kubeletconfiginternal.ImagePullIntent{
				{Image: "test.repo/org1/project1:tag"},
			}),
			wantRecordedMethods: []string{"WriteImagePullIntent"},
		},
		{
			name: "write to delegate returns error",
			delegate: &recordingPullRecordsAccessor{
				delegate: newTestPullRecordsAccessor(nil, nil).withNextGetWriteError(fmt.Errorf("write error")),
			},
			initialCache:        map[string]*kubeletconfiginternal.ImagePullIntent{},
			inputImage:          "test.repo/org1/project1:tag",
			wantCacheIntents:    map[string]*kubeletconfiginternal.ImagePullIntent{},
			wantErr:             fmt.Errorf("write error"),
			wantRecordedMethods: []string{"WriteImagePullIntent"},
		},
		{
			name: "write to cache that's already populated",
			delegate: &recordingPullRecordsAccessor{
				delegate: newTestPullRecordsAccessor(nil, nil),
			},
			initialCache: pullIntentsToMap(testPullIntents()),
			inputImage:   "test.repo/org1/newimage:tag",
			wantCacheIntents: pullIntentsToMap(
				append(testPullIntents(), &kubeletconfiginternal.ImagePullIntent{Image: "test.repo/org1/newimage:tag"}),
			),
			wantRecordedMethods: []string{"WriteImagePullIntent"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &cachedPullRecordsAccessor{
				delegate:     tt.delegate,
				intentsMutex: &sync.RWMutex{},
				intents:      tt.initialCache,
			}
			if err := c.WriteImagePullIntent(tt.inputImage); !cmpErrorStrings(err, tt.wantErr) {
				t.Errorf("cachedPullRecordsAccessor.WriteImagePullIntent() error = %v, wantErr %v", err, tt.wantErr)
			}

			if !reflect.DeepEqual(c.intents, tt.wantCacheIntents) {
				t.Errorf("cachedPullRecordsAccessor.ImagePullIntentExists() cache state = %v, want %v", c.intents, tt.wantCacheIntents)
			}

			if !reflect.DeepEqual(tt.delegate.methodsCalled, tt.wantRecordedMethods) {
				t.Errorf("cachedPullRecordsAccessor.WriteImagePullIntent() recorded methods = %v, want %v", tt.delegate.methodsCalled, tt.wantRecordedMethods)
			}
		})
	}
}

func Test_cachedPullRecordsAccessor_WriteImagePulledRecord(t *testing.T) {
	tests := []struct {
		name                   string
		delegate               *recordingPullRecordsAccessor
		initialCache           map[string]*kubeletconfiginternal.ImagePulledRecord
		inputRecord            *kubeletconfiginternal.ImagePulledRecord
		wantErr                error
		wantCachePulledRecords map[string]*kubeletconfiginternal.ImagePulledRecord
		wantRecordedMethods    []string
	}{
		{
			name: "successful write through an empty cache to delegate",
			delegate: &recordingPullRecordsAccessor{
				delegate: newTestPullRecordsAccessor(nil, nil),
			},
			initialCache: map[string]*kubeletconfiginternal.ImagePulledRecord{},
			inputRecord: &kubeletconfiginternal.ImagePulledRecord{
				ImageRef: "someref-1",
			},
			wantCachePulledRecords: pulledRecordsToMap([]*kubeletconfiginternal.ImagePulledRecord{
				{ImageRef: "someref-1"},
			}),
			wantRecordedMethods: []string{"WriteImagePulledRecord"},
		},
		{
			name: "rewrite cached pulled record, write through to delegate",
			delegate: &recordingPullRecordsAccessor{
				delegate: newTestPullRecordsAccessor(nil, nil),
			},
			initialCache: map[string]*kubeletconfiginternal.ImagePulledRecord{
				"someref-1": {
					ImageRef: "someref-1",
					CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
						"test.repo/org1/project1": {
							NodePodsAccessible: true,
						},
					},
				},
			},
			inputRecord: &kubeletconfiginternal.ImagePulledRecord{
				ImageRef: "someref-1",
				CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
					"test.repo/org1/project1": {
						NodePodsAccessible: true,
					},
					"test.repo/org2/project1": {
						KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
							{Namespace: "ns1", Name: "secret1", UID: "uid1", CredentialHash: "somehash"},
						},
					},
				},
			},
			wantCachePulledRecords: pulledRecordsToMap([]*kubeletconfiginternal.ImagePulledRecord{
				{
					ImageRef: "someref-1",
					CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
						"test.repo/org1/project1": {
							NodePodsAccessible: true,
						},
						"test.repo/org2/project1": {
							KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
								{Namespace: "ns1", Name: "secret1", UID: "uid1", CredentialHash: "somehash"},
							},
						},
					},
				},
			}),
			wantRecordedMethods: []string{"WriteImagePulledRecord"},
		},
		{
			name: "write even though the same pulled record is already in the cache",
			delegate: &recordingPullRecordsAccessor{
				delegate: newTestPullRecordsAccessor(nil, nil),
			},
			initialCache: map[string]*kubeletconfiginternal.ImagePulledRecord{
				"someref-1": {ImageRef: "someref-1"},
			},
			inputRecord: &kubeletconfiginternal.ImagePulledRecord{
				ImageRef: "someref-1",
			},
			wantCachePulledRecords: pulledRecordsToMap([]*kubeletconfiginternal.ImagePulledRecord{
				{ImageRef: "someref-1"},
			}),
			wantRecordedMethods: []string{"WriteImagePulledRecord"},
		},
		{
			name: "write to delegate returns error",
			delegate: &recordingPullRecordsAccessor{
				delegate: newTestPullRecordsAccessor(nil, nil).withNextGetWriteError(fmt.Errorf("write error")),
			},
			initialCache: map[string]*kubeletconfiginternal.ImagePulledRecord{},
			inputRecord: &kubeletconfiginternal.ImagePulledRecord{
				ImageRef: "someref-1",
			},
			wantCachePulledRecords: map[string]*kubeletconfiginternal.ImagePulledRecord{},
			wantErr:                fmt.Errorf("write error"),
			wantRecordedMethods:    []string{"WriteImagePulledRecord"},
		},
		{
			name: "write to cache that's already populated",
			delegate: &recordingPullRecordsAccessor{
				delegate: newTestPullRecordsAccessor(nil, nil),
			},
			initialCache: pulledRecordsToMap(testPulledRecords()),
			inputRecord: &kubeletconfiginternal.ImagePulledRecord{
				ImageRef: "newref",
			},
			wantCachePulledRecords: pulledRecordsToMap(
				append(testPulledRecords(), &kubeletconfiginternal.ImagePulledRecord{ImageRef: "newref"}),
			),
			wantRecordedMethods: []string{"WriteImagePulledRecord"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &cachedPullRecordsAccessor{
				delegate:           tt.delegate,
				pulledRecordsMutex: &sync.RWMutex{},
				pulledRecords:      tt.initialCache,
			}
			if err := c.WriteImagePulledRecord(tt.inputRecord); !cmpErrorStrings(err, tt.wantErr) {
				t.Errorf("cachedPullRecordsAccessor.WriteImagePulledRecord() error = %v, wantErr %v", err, tt.wantErr)
			}

			if !reflect.DeepEqual(c.pulledRecords, tt.wantCachePulledRecords) {
				t.Errorf("cachedPullRecordsAccessor.WriteImagePulledRecord() cache state = %v, want %v", c.pulledRecords, tt.wantCachePulledRecords)
			}

			if !reflect.DeepEqual(tt.delegate.methodsCalled, tt.wantRecordedMethods) {
				t.Errorf("cachedPullRecordsAccessor.WriteImagePulledRecord() recorded methods = %v, want %v", tt.delegate.methodsCalled, tt.wantRecordedMethods)
			}
		})
	}
}

func Test_cachedPullRecordsAccessor_DeleteImagePullIntent(t *testing.T) {
	tests := []struct {
		name         string
		delegate     *testPullRecordsAccessor
		initialCache map[string]*kubeletconfiginternal.ImagePullIntent
		inputImage   string
		wantErr      error
		wantIntents  map[string]*kubeletconfiginternal.ImagePullIntent
	}{
		{
			name:         "delete an existing intent from the cache",
			delegate:     newTestPullRecordsAccessor(testPullIntents(), nil),
			initialCache: pullIntentsToMap(testPullIntents()),
			inputImage:   "test.repo/org1/project1:tag",
			wantIntents:  deleteFromStringMap(pullIntentsToMap(testPullIntents()), "test.repo/org1/project1:tag"),
		},
		{
			name:         "attempt to delete a non-existent intent from the cache",
			delegate:     newTestPullRecordsAccessor(testPullIntents(), nil),
			initialCache: pullIntentsToMap(testPullIntents()),
			inputImage:   "test.repo/org1/doesnotexist:tag",
			wantIntents:  pullIntentsToMap(testPullIntents()),
		},
		{
			name:         "error deleting an intent from delegate",
			delegate:     newTestPullRecordsAccessor(testPullIntents(), nil).withNextGetWriteError(fmt.Errorf("error deleting intent")),
			initialCache: pullIntentsToMap(testPullIntents()),
			inputImage:   "test.repo/org1/project1:tag",
			wantIntents:  pullIntentsToMap(testPullIntents()),
			wantErr:      fmt.Errorf("error deleting intent"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &cachedPullRecordsAccessor{
				delegate:     tt.delegate,
				intentsMutex: &sync.RWMutex{},
				intents:      tt.initialCache,
			}
			if err := c.DeleteImagePullIntent(tt.inputImage); !cmpErrorStrings(err, tt.wantErr) {
				t.Errorf("cachedPullRecordsAccessor.DeleteImagePullIntent() error = %v, wantErr %v", err, tt.wantErr)
			}

			if !reflect.DeepEqual(c.intents, tt.wantIntents) {
				t.Errorf("cachedPullRecordsAccessor.DeleteImagePullIntent() cache state = %v, want %v", c.intents, tt.wantIntents)
			}

			if !reflect.DeepEqual(tt.delegate.intents, tt.wantIntents) {
				t.Errorf("cachedPullRecordsAccessor.DeleteImagePullIntent() delegate intents = %v, want %v", tt.delegate.intents, tt.wantIntents)
			}
		})
	}
}

func Test_cachedPullRecordsAccessor_DeleteImagePulledRecord(t *testing.T) {
	tests := []struct {
		name              string
		delegate          *testPullRecordsAccessor
		initialCache      map[string]*kubeletconfiginternal.ImagePulledRecord
		imageRef          string
		wantErr           error
		wantPulledRecords map[string]*kubeletconfiginternal.ImagePulledRecord
	}{
		{
			name:              "delete an existing pulled record from the cache",
			delegate:          newTestPullRecordsAccessor(nil, testPulledRecords()),
			initialCache:      pulledRecordsToMap(testPulledRecords()),
			imageRef:          "ref-22",
			wantPulledRecords: deleteFromStringMap(pulledRecordsToMap(testPulledRecords()), "ref-22"),
		},
		{
			name:              "attempt to delete a non-existent imagePulledRecord from the cache",
			delegate:          newTestPullRecordsAccessor(nil, testPulledRecords()),
			initialCache:      pulledRecordsToMap(testPulledRecords()),
			imageRef:          "non-existent-ref",
			wantPulledRecords: pulledRecordsToMap(testPulledRecords()),
		},
		{
			name:              "error deleting an imagePulledRecord from delegate",
			delegate:          newTestPullRecordsAccessor(nil, testPulledRecords()).withNextGetWriteError(fmt.Errorf("error deleting imagePulledRecord")),
			initialCache:      pulledRecordsToMap(testPulledRecords()),
			imageRef:          "ref-22",
			wantPulledRecords: pulledRecordsToMap(testPulledRecords()),
			wantErr:           fmt.Errorf("error deleting imagePulledRecord"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &cachedPullRecordsAccessor{
				delegate:           tt.delegate,
				pulledRecordsMutex: &sync.RWMutex{},
				pulledRecords:      tt.initialCache,
			}
			if err := c.DeleteImagePulledRecord(tt.imageRef); !cmpErrorStrings(err, tt.wantErr) {
				t.Errorf("cachedPullRecordsAccessor.DeleteImagePulledRecord() error = %v, wantErr %v", err, tt.wantErr)
			}

			if !reflect.DeepEqual(c.pulledRecords, tt.wantPulledRecords) {
				t.Errorf("cachedPullRecordsAccessor.DeleteImagePulledRecord() cache state = %v, want %v", c.pulledRecords, tt.wantPulledRecords)
			}

			if !reflect.DeepEqual(tt.delegate.pulledRecords, tt.wantPulledRecords) {
				t.Errorf("cachedPullRecordsAccessor.DeleteImagePulledRecord() delegate imagePulledRecords = %v, want %v", tt.delegate.pulledRecords, tt.wantPulledRecords)
			}
		})
	}
}

func Test_cachedPullRecordsAccessor_GetImagePulledRecord(t *testing.T) {

	tests := []struct {
		name                   string
		delegate               PullRecordsAccessor
		cachedPulledRecords    map[string]*kubeletconfiginternal.ImagePulledRecord
		imageRef               string
		wantPulledRecord       *kubeletconfiginternal.ImagePulledRecord
		wantExists             bool
		wantErr                error
		wantCachePulledRecords map[string]*kubeletconfiginternal.ImagePulledRecord
	}{
		{
			name:                   "pulled record exists in cache",
			delegate:               newTestPullRecordsAccessor(nil, nil),
			cachedPulledRecords:    pulledRecordsToMap(testPulledRecords()),
			imageRef:               "ref-009",
			wantPulledRecord:       pulledRecordsToMap(testPulledRecords())["ref-009"],
			wantExists:             true,
			wantCachePulledRecords: pulledRecordsToMap(testPulledRecords()),
		},
		{
			name:                   "pulled record exists in delegate, cache gets updated",
			delegate:               newTestPullRecordsAccessor(nil, testPulledRecords()),
			cachedPulledRecords:    deleteFromStringMap(pulledRecordsToMap(testPulledRecords()), "ref-22"),
			imageRef:               "ref-22",
			wantPulledRecord:       pulledRecordsToMap(testPulledRecords())["ref-22"],
			wantExists:             true,
			wantCachePulledRecords: pulledRecordsToMap(testPulledRecords()),
		},
		{
			name:                   "pulled record does not exist",
			delegate:               newTestPullRecordsAccessor(nil, nil),
			cachedPulledRecords:    deleteFromStringMap(pulledRecordsToMap(testPulledRecords()), "ref-87"),
			imageRef:               "ref-87",
			wantPulledRecord:       nil,
			wantExists:             false,
			wantCachePulledRecords: deleteFromStringMap(pulledRecordsToMap(testPulledRecords()), "ref-87"),
		},
		{
			name:                   "pulled record not cached and not in delegate, delegate returns error on read",
			delegate:               newTestPullRecordsAccessor(nil, nil).withNextGetWriteError(fmt.Errorf("error reading pulled record")),
			cachedPulledRecords:    map[string]*kubeletconfiginternal.ImagePulledRecord{},
			imageRef:               "doesnotexist",
			wantPulledRecord:       nil,
			wantExists:             false,
			wantErr:                fmt.Errorf("error reading pulled record"),
			wantCachePulledRecords: map[string]*kubeletconfiginternal.ImagePulledRecord{},
		},
		{
			name:                   "pulled record not cached, exists in delegate, delegate returns error",
			delegate:               newTestPullRecordsAccessor(nil, testPulledRecords()).withNextGetWriteError(fmt.Errorf("error reading pulled record")),
			cachedPulledRecords:    map[string]*kubeletconfiginternal.ImagePulledRecord{},
			imageRef:               "ref-3",
			wantPulledRecord:       nil,
			wantExists:             true,
			wantErr:                fmt.Errorf("error reading pulled record"),
			wantCachePulledRecords: map[string]*kubeletconfiginternal.ImagePulledRecord{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &cachedPullRecordsAccessor{
				delegate:           tt.delegate,
				pulledRecordsMutex: &sync.RWMutex{},
				pulledRecords:      tt.cachedPulledRecords,
			}
			gotPulledRecord, gotExists, err := c.GetImagePulledRecord(tt.imageRef)
			if !cmpErrorStrings(err, tt.wantErr) {
				t.Errorf("cachedPullRecordsAccessor.GetImagePulledRecord() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotPulledRecord, tt.wantPulledRecord) {
				t.Errorf("cachedPullRecordsAccessor.GetImagePulledRecord() gotPulledRecord = %v, want %v", gotPulledRecord, tt.wantPulledRecord)
			}
			if gotExists != tt.wantExists {
				t.Errorf("cachedPullRecordsAccessor.GetImagePulledRecord() gotExists = %v, want %v", gotExists, tt.wantExists)
			}

			if !reflect.DeepEqual(c.pulledRecords, tt.wantCachePulledRecords) {
				t.Errorf("cachedPullRecordsAccessor.GetImagePulledRecord() final cache state = %v, want %v", c.pulledRecords, tt.wantCachePulledRecords)
			}
		})
	}
}

func testPullIntents() []*kubeletconfiginternal.ImagePullIntent {
	return []*kubeletconfiginternal.ImagePullIntent{
		{Image: "test.repo/org1/project2:tag"},
		{Image: "test.repo/org1/project1:tag"},
		{Image: "test.repo/org1/project3:tag"},
		{Image: "test.repo/org2/project1:tag"},
	}
}

func testPulledRecords() []*kubeletconfiginternal.ImagePulledRecord {
	return []*kubeletconfiginternal.ImagePulledRecord{
		{
			ImageRef: "ref-87",
			CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
				"image.some/org1/imagex": {
					NodePodsAccessible: true,
				},
			},
		},
		{
			ImageRef: "ref-3",
			CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
				"image.some/org2/image": {
					NodePodsAccessible: true,
				},
			},
		},
		{ImageRef: "ref-22",
			CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
				"image.some/org1/image-2": {
					KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
						{Namespace: "ns1", Name: "secret1", UID: "uid1", CredentialHash: "somehash"},
					},
				},
			},
		},
		{ImageRef: "ref-17",
			CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
				"image.some/org1/image-3": {
					NodePodsAccessible: true,
				},
			},
		},
		{ImageRef: "ref-009",
			CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
				"image.some/org3/image": {
					NodePodsAccessible: true,
				},
			},
		},
	}
}

func pullIntentsToMap(intents []*kubeletconfiginternal.ImagePullIntent) map[string]*kubeletconfiginternal.ImagePullIntent {
	ret := make(map[string]*kubeletconfiginternal.ImagePullIntent)
	for _, intent := range intents {
		ret[intent.Image] = intent
	}
	return ret
}

func pulledRecordsToMap(records []*kubeletconfiginternal.ImagePulledRecord) map[string]*kubeletconfiginternal.ImagePulledRecord {
	ret := make(map[string]*kubeletconfiginternal.ImagePulledRecord)
	for _, record := range records {
		ret[record.ImageRef] = record
	}
	return ret
}

func testPullIntentsSorted() []*kubeletconfiginternal.ImagePullIntent {
	intents := testPullIntents()
	slices.SortFunc(intents, pullIntentsCmp)
	return intents
}

func testPulledRecordsSorted() []*kubeletconfiginternal.ImagePulledRecord {
	records := testPulledRecords()
	slices.SortFunc(records, pulledRecordsCmp)
	return records
}

func cmpErrorStrings(a, b error) bool {
	if a == nil && b == nil {
		return true
	} else if a == nil || b == nil {
		return false
	}

	return strings.Compare(a.Error(), b.Error()) == 0
}

func deleteFromStringMap[V any](m map[string]V, key string) map[string]V {
	delete(m, key)
	return m
}
