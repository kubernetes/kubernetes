/*
Copyright 2017 The Kubernetes Authors.

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

package nodeconfig

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	kuberuntime "k8s.io/apimachinery/pkg/runtime"
	api "k8s.io/kubernetes/pkg/api"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
)

// checkpointExists returns true if checkpoint for the config source identified by `uid` exists on disk.
// If the existence of a checkpoint cannot be determined due to filesystem issues, a fatal error occurs.
func (cc *NodeConfigController) checkpointExists(uid string) bool {
	ok, err := cc.dirExists(filepath.Join(checkpointsDir, uid))
	if err != nil {
		fatalf("failed to determine whether checkpoint %q exists, error: %v", uid, err)
	}
	return ok
}

// loadCheckpoint loads the checkpoint at `cc.configDir/relPath`, which is expected to be a checkpoint directory.
// If the checkpoint directory does not exist or if data cannot be retrieved from the filesystem,
// a fatal error will occur.
// If the data cannot be decoded and converted to a supported config source type, returns an error.
// This may indicate a failure to completely save the checkpoint. You may want to attempt a re-download in this scenario.
// If loading succeeds, returns a `verifiable` (see verify.go). This interface can be used to verify the integrity of
// the loaded checkpoint.
func (cc *NodeConfigController) loadCheckpoint(relPath string) (verifiable, error) {
	path := filepath.Join(cc.configDir, relPath)
	infof("loading configuration from %q", path)

	// find the checkpoint file(s)
	files, err := ioutil.ReadDir(path)
	if err != nil {
		fatalf("failed to enumerate checkpoint files in dir %q, error: %v", path, err)
	} else if len(files) == 0 {
		return nil, fmt.Errorf("no checkpoint files in dir %q, but there should be at least one", path)
	}

	// TODO(mtaufen): for now, we only have one file per checkpoint (a serialized API object, e.g. a ConfigMap); if this ever changes we will need to extend this
	file := files[0]
	filePath := filepath.Join(path, file.Name())
	b, err := ioutil.ReadFile(filePath)
	if err != nil {
		fatalf("failed to read checkpoint file %q, error: %v", filePath, err)
	}

	// decode the checkpoint file
	obj, err := kuberuntime.Decode(api.Codecs.UniversalDecoder(), b)
	if err != nil {
		return nil, fmt.Errorf("failed to decode checkpoint file %q, error: %v", filePath, err)
	}

	// TODO(mtaufen): for now we assume we are trying to load a ConfigMap, but we may need to eventually be generic to the type

	// convert it to the external ConfigMap type, so we're consistently working with the external type outside of the on-disk representation
	cm := &apiv1.ConfigMap{}
	err = api.Scheme.Convert(obj, cm, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to convert decoded object into a v1 ConfigMap, error: %v", err)
	}

	return &verifiableConfigMap{cm: cm}, nil
}

// checkpointConfigMap saves `cm` to disk, at the appropriate checkpoint path.
// If saving the checkpoint fails, returns an error.
// If cleanup after a failed save fails, also returns an error, so the caller can decide what to do.
func (cc *NodeConfigController) checkpointConfigMap(cm *apiv1.ConfigMap) (reterr error) {
	uidPath := filepath.Join(cc.configDir, checkpointsDir, string(cm.ObjectMeta.UID))
	err := os.Mkdir(uidPath, defaultPerm)
	if err != nil {
		return fmt.Errorf("failed to checkpoint ConfigMap with UID %s, err: %v", cm.ObjectMeta.UID, err)
	}

	// defer cleanup function now that we have something to clean up (we just created a dir)
	defer func() {
		if reterr != nil {
			// clean up the checkpoint dir
			rmerr := os.RemoveAll(uidPath)
			if rmerr != nil {
				reterr = fmt.Errorf("failed to checkpoint ConfigMap with UID %s, error: %v; failed to clean up checkpoint dir, error: %v", cm.ObjectMeta.UID, reterr, rmerr)
			}
			reterr = fmt.Errorf("failed to checkpoint ConfigMap with UID %s, error: %v", cm.ObjectMeta.UID, reterr)
		}
	}()

	// checkpoint the configmap object we got
	filePath := filepath.Join(uidPath, cm.Name)

	// serialize to json
	mediaType := "application/json"
	info, ok := kuberuntime.SerializerInfoForMediaType(api.Codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		reterr = fmt.Errorf("unsupported media type %q", mediaType)
		return
	}

	versions := api.Registry.EnabledVersionsForGroup(apiv1.GroupName)
	if len(versions) == 0 {
		reterr = fmt.Errorf("no enabled versions for group %q", apiv1.GroupName)
		return
	}

	// the "best" version supposedly comes first in the list returned from api.Registry.EnabledVersionsForGroup
	encoder := api.Codecs.EncoderForVersion(info.Serializer, versions[0])
	b, reterr := kuberuntime.Encode(encoder, cm)
	if reterr != nil {
		return
	}

	// save the file
	reterr = ioutil.WriteFile(filePath, b, defaultPerm)
	if reterr != nil {
		return
	}
	return
}
