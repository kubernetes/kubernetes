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
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/klog/v2"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletconfigint1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/config/v1alpha1"
	kubeletconfigint1beta1 "k8s.io/kubernetes/pkg/kubelet/apis/config/v1beta1"
)

const (
	cacheFilesSHA256Prefix = "sha256-"
	tmpFilesSuffix         = ".tmp"
)

var encodeVersion = kubeletconfigv1beta1.SchemeGroupVersion

var _ PullRecordsAccessor = &fsPullRecordsAccessor{}

// fsPullRecordsAccessor uses the filesystem to read/write ImagePullIntent/ImagePulledRecord
// records.
type fsPullRecordsAccessor struct {
	pullingDir string
	pulledDir  string

	encoder runtime.Encoder
	decoder runtime.Decoder
}

// NewFSPullRecordsAccessor returns an accessor for the ImagePullIntent/ImagePulledRecord
// records with a filesystem as the backing database.
func NewFSPullRecordsAccessor(kubeletDir string) (PullRecordsAccessor, error) {
	kubeletConfigEncoder, kubeletConfigDecoder, err := createKubeletConfigSchemeEncoderDecoder()
	if err != nil {
		return nil, err
	}

	accessor := &fsPullRecordsAccessor{
		pullingDir: filepath.Join(kubeletDir, "image_manager", "pulling"),
		pulledDir:  filepath.Join(kubeletDir, "image_manager", "pulled"),

		encoder: kubeletConfigEncoder,
		decoder: kubeletConfigDecoder,
	}

	if err := os.MkdirAll(accessor.pullingDir, 0700); err != nil {
		return nil, err
	}

	if err := os.MkdirAll(accessor.pulledDir, 0700); err != nil {
		return nil, err
	}

	accessor.recordsVersionMigration()
	return NewMeteringRecordsAccessor(accessor, fsPullIntentsSize, fsPulledRecordsSize), nil
}

func (f *fsPullRecordsAccessor) recordsVersionMigration() {
	err := processDirFiles(f.pullingDir,
		func(filePath string, fileContent []byte) error {
			intent, isCurrentVersion, err := decodeIntent(f.decoder, fileContent)
			if err != nil {
				return fmt.Errorf("failed to decode ImagePullIntent from file %q: %w", filePath, err)
			}
			if !isCurrentVersion {
				if err := f.WriteImagePullIntent(intent.Image); err != nil {
					return fmt.Errorf("failed to migrate ImagePullIntent for image %q to the current version: %w", intent.Image, err)
				}
			}
			return nil
		})
	if err != nil {
		klog.ErrorS(err, "Error migrating image pull intents")
	}
	err = processDirFiles(f.pulledDir,
		func(filePath string, fileContent []byte) error {
			pullRecord, isCurrentVersion, err := decodePulledRecord(f.decoder, fileContent)
			if err != nil {
				return fmt.Errorf("failed to decode ImagePulledRecord from file %q: %w", filePath, err)
			}
			if !isCurrentVersion {
				if err := f.WriteImagePulledRecord(pullRecord); err != nil {
					return fmt.Errorf("failed to migrate ImagePulledRecord for image ref %q to the current version: %w", pullRecord.ImageRef, err)
				}
			}
			return nil
		})

	if err != nil {
		klog.ErrorS(err, "Error migrating image pulled records")
	}
}

func (f *fsPullRecordsAccessor) WriteImagePullIntent(image string) error {
	intent := kubeletconfiginternal.ImagePullIntent{
		Image: image,
	}

	intentBytes := bytes.NewBuffer([]byte{})
	if err := f.encoder.Encode(&intent, intentBytes); err != nil {
		return err
	}

	return writeFile(f.pullingDir, cacheFilename(image), intentBytes.Bytes())
}

func (f *fsPullRecordsAccessor) ListImagePullIntents() ([]*kubeletconfiginternal.ImagePullIntent, error) {
	var intents []*kubeletconfiginternal.ImagePullIntent
	// walk the pulling directory for any pull intent records
	err := processDirFiles(f.pullingDir,
		func(filePath string, fileContent []byte) error {
			intent, _, err := decodeIntent(f.decoder, fileContent)
			if err != nil {
				return fmt.Errorf("failed to deserialize content of file %q into ImagePullIntent: %w", filePath, err)
			}
			intents = append(intents, intent)

			return nil
		})
	return intents, err
}

func (f *fsPullRecordsAccessor) ImagePullIntentExists(image string) (bool, error) {
	intentRecordPath := filepath.Join(f.pullingDir, cacheFilename(image))
	intentBytes, err := os.ReadFile(intentRecordPath)
	if os.IsNotExist(err) {
		return false, nil
	} else if err != nil {
		return false, err
	}

	intent, _, err := decodeIntent(f.decoder, intentBytes)
	if err != nil {
		return false, err
	}

	return intent.Image == image, nil
}

func (f *fsPullRecordsAccessor) DeleteImagePullIntent(image string) error {
	err := os.Remove(filepath.Join(f.pullingDir, cacheFilename(image)))
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

func (f *fsPullRecordsAccessor) GetImagePulledRecord(imageRef string) (*kubeletconfiginternal.ImagePulledRecord, bool, error) {
	recordBytes, err := os.ReadFile(filepath.Join(f.pulledDir, cacheFilename(imageRef)))
	if os.IsNotExist(err) {
		return nil, false, nil
	} else if err != nil {
		return nil, false, err
	}

	pulledRecord, _, err := decodePulledRecord(f.decoder, recordBytes)
	if err != nil {
		return nil, true, err
	}
	if pulledRecord.ImageRef != imageRef {
		return nil, false, nil
	}
	return pulledRecord, true, err
}

func (f *fsPullRecordsAccessor) ListImagePulledRecords() ([]*kubeletconfiginternal.ImagePulledRecord, error) {
	var pullRecords []*kubeletconfiginternal.ImagePulledRecord
	err := processDirFiles(f.pulledDir,
		func(filePath string, fileContent []byte) error {
			pullRecord, _, err := decodePulledRecord(f.decoder, fileContent)
			if err != nil {
				return fmt.Errorf("failed to deserialize content of file %q into ImagePulledRecord: %w", filePath, err)
			}
			pullRecords = append(pullRecords, pullRecord)
			return nil
		})

	return pullRecords, err
}

func (f *fsPullRecordsAccessor) WriteImagePulledRecord(pulledRecord *kubeletconfiginternal.ImagePulledRecord) error {
	recordBytes := bytes.NewBuffer([]byte{})
	if err := f.encoder.Encode(pulledRecord, recordBytes); err != nil {
		return fmt.Errorf("failed to serialize ImagePulledRecord: %w", err)
	}

	return writeFile(f.pulledDir, cacheFilename(pulledRecord.ImageRef), recordBytes.Bytes())
}

func (f *fsPullRecordsAccessor) DeleteImagePulledRecord(imageRef string) error {
	err := os.Remove(filepath.Join(f.pulledDir, cacheFilename(imageRef)))
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

func (f *fsPullRecordsAccessor) intentsSize() (uint, error) {
	intentsCount, err := countCacheFiles(f.pullingDir)
	if err != nil {
		return 0, err
	}
	return intentsCount, nil
}

func (f *fsPullRecordsAccessor) pulledRecordsSize() (uint, error) {
	pulledRecordsCount, err := countCacheFiles(f.pulledDir)
	if err != nil {
		return 0, err
	}
	return pulledRecordsCount, nil
}

func countCacheFiles(dirName string) (uint, error) {
	const readBatch = 20

	var cacheFilesCount uint
	dir, err := os.Open(dirName)
	if err != nil {
		return 0, fmt.Errorf("failed to open directory %q: %w", dirName, err)
	}
	// ignoring close error on a readonly open should be safe
	defer func() { _ = dir.Close() }()

	for {
		entries, err := dir.ReadDir(readBatch)
		for _, entry := range entries {
			if !isValidCacheFile(entry) {
				continue
			}
			cacheFilesCount++
		}
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return 0, fmt.Errorf("failed to list all entries in directory %q: %w", dirName, err)
		}
	}
	return cacheFilesCount, nil
}

func cacheFilename(image string) string {
	return fmt.Sprintf("%s%x", cacheFilesSHA256Prefix, sha256.Sum256([]byte(image)))
}

// isValidCacheFile returns true if the info doesn't point to a directory and
// the filename matches the expectation for a valid, non-temporary cache file.
func isValidCacheFile(fileInfo os.DirEntry) bool {
	if fileInfo.IsDir() {
		return false
	}

	filename := fileInfo.Name()
	return strings.HasPrefix(filename, cacheFilesSHA256Prefix) && !strings.HasSuffix(filename, tmpFilesSuffix)
}

// writeFile writes `content` to the file with name `filename` in directory `dir`.
// It assures write atomicity by creating a temporary file first and only after
// a successful write, it move the temp file in place of the target.
func writeFile(dir, filename string, content []byte) error {
	// create target folder if it does not exists yet
	if err := os.MkdirAll(dir, 0700); err != nil {
		return fmt.Errorf("failed to create directory %q: %w", dir, err)
	}

	targetPath := filepath.Join(dir, filename)
	tmpPath := targetPath + tmpFilesSuffix
	if err := os.WriteFile(tmpPath, content, 0600); err != nil {
		_ = os.Remove(tmpPath) // attempt a delete in case the file was at least partially written
		return fmt.Errorf("failed to create temporary file %q: %w", tmpPath, err)
	}

	if err := os.Rename(tmpPath, targetPath); err != nil {
		_ = os.Remove(tmpPath) // attempt a cleanup
		return err
	}
	return nil
}

// processDirFiles reads files in a given directory and peforms `fileAction` action on those.
func processDirFiles(dirName string, fileAction func(filePath string, fileContent []byte) error) error {
	var walkErrors []error
	err := filepath.WalkDir(dirName, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			walkErrors = append(walkErrors, err)
			return nil
		}

		if path == dirName {
			return nil
		}

		if d.IsDir() {
			return filepath.SkipDir
		}

		// skip files we didn't write or .tmp files
		if filename := d.Name(); !strings.HasPrefix(filename, cacheFilesSHA256Prefix) || strings.HasSuffix(filename, tmpFilesSuffix) {
			return nil
		}

		fileContent, err := os.ReadFile(path)
		if err != nil {
			walkErrors = append(walkErrors, fmt.Errorf("failed to read %q: %w", path, err))
			return nil
		}

		if err := fileAction(path, fileContent); err != nil {
			walkErrors = append(walkErrors, err)
			return nil
		}

		return nil
	})
	if err != nil {
		walkErrors = append(walkErrors, err)
	}

	return utilerrors.NewAggregate(walkErrors)
}

// createKubeletCOnfigSchemeEncoderDecoder creates strict-encoding encoder and
// decoder for the internal and alpha kubelet config APIs.
func createKubeletConfigSchemeEncoderDecoder() (runtime.Encoder, runtime.Decoder, error) {
	codecs, info, err := getKubeletConfigSerializerInfo()
	if err != nil {
		return nil, nil, err
	}
	return codecs.EncoderForVersion(info.Serializer, encodeVersion), codecs.UniversalDecoder(), nil
}

func getKubeletConfigSerializerInfo() (*serializer.CodecFactory, *runtime.SerializerInfo, error) {
	const mediaType = runtime.ContentTypeJSON

	scheme := runtime.NewScheme()
	if err := kubeletconfigint1beta1.AddToScheme(scheme); err != nil {
		return nil, nil, err
	}
	if err := kubeletconfigint1alpha1.AddToScheme(scheme); err != nil {
		return nil, nil, err
	}
	if err := kubeletconfiginternal.AddToScheme(scheme); err != nil {
		return nil, nil, err
	}

	// use the strict scheme to fail on unknown fields
	codecs := serializer.NewCodecFactory(scheme, serializer.EnableStrict)

	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return nil, nil, fmt.Errorf("unable to locate encoder -- %q is not a supported media type", mediaType)
	}
	return &codecs, &info, nil
}

// decodeIntent decodes the image pull intent and converts it to the internal API version.
// The decoder must be aware of the schemas available for the kubelet config API.
//
// The returned object is an internal representation of the image pull intent and a bool
// that indicates whether the image pull intent appears in the latest known version of the kubelet config API.
func decodeIntent(d runtime.Decoder, objBytes []byte) (*kubeletconfiginternal.ImagePullIntent, bool, error) {
	obj, gvk, err := d.Decode(objBytes, nil, nil)
	if err != nil {
		return nil, false, err
	}

	intentObj, ok := obj.(*kubeletconfiginternal.ImagePullIntent)
	if !ok {
		return nil, false, fmt.Errorf("failed to convert object to *ImagePullIntent: %T", obj)
	}

	isLatestVersion := gvk.Version == encodeVersion.Version
	return intentObj, isLatestVersion, nil
}

// decodePulledRecord decodes the pulled record and converts it to the internal API version.
// The decoder must be aware of the schemas available for the kubelet config API.
//
// The returned object is an internal representation of the pulled record and a bool
// that indicates whether the record appears in the latest known version of the kubelet config API.
func decodePulledRecord(d runtime.Decoder, objBytes []byte) (*kubeletconfiginternal.ImagePulledRecord, bool, error) {
	obj, gvk, err := d.Decode(objBytes, nil, nil)
	if err != nil {
		return nil, false, err
	}

	pulledRecord, ok := obj.(*kubeletconfiginternal.ImagePulledRecord)
	if !ok {
		return nil, false, fmt.Errorf("failed to convert object to *ImagePulledRecord: %T", obj)
	}

	isLatestVersion := gvk.Version == encodeVersion.Version
	return pulledRecord, isLatestVersion, nil
}
