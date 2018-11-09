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

package csi

import (
	"context"
	"errors"

	csipb "github.com/container-storage-interface/spec/lib/go/csi/v0"
	"github.com/golang/glog"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	api "k8s.io/api/core/v1"

	"k8s.io/apimachinery/pkg/util/wait"
)

// inlineProvision will request the CSI driver to create a new volume.
// It will attempt to lookup referenced PVC, if one is provided, otherwise
// it will create a volume with default size of zero.
// Returns csi.Volume or error if failure.
func (c *csiAttacher) inlineProvision(volSpecName, namespace string, volSource *api.CSIVolumeSource) (*csipb.Volume, error) {
	glog.V(4).Info(log("mounter.inlineProvision called for volume %s", volSpecName))

	if volSource == nil {
		return nil, errors.New("missing inline VolumeSource")
	}

	if volSource == nil {
		return nil, errors.New("missing inline CSIVolumeSource")
	}

	// get controller pub secrets
	secrets := map[string]string{}
	if volSource.ControllerPublishSecretRef != nil {
		/*		name := csiSource.ControllerPublishSecretRef.Name
				sec, err := volutil.GetSecretForPod(c.pod, name, c.k8s)
				if err != nil {
					return nil, err
				}
				secrets = sec */
	}

	// make controller pub request
	var volume *csipb.Volume
	err := wait.ExponentialBackoff(defaultBackoff(), func() (bool, error) {
		ctx, cancel := context.WithTimeout(context.Background(), csiDefaultTimeout)
		defer cancel()
		vol, createErr := c.csiClient.CreateVolume(ctx, volSpecName, int64(0), secrets)
		if createErr == nil {
			volume = vol
			return true, nil
		}

		// can we recover
		if status, ok := status.FromError(createErr); ok {
			if status.Code() == codes.DeadlineExceeded {
				// CreateVolume timed out, give it another chance to complete
				glog.Warningf("Mounter.inlineProvision CreateVolume timed out, operation will be retried")
				return false, nil
			}
		}

		// CreateVolume failed , no reason to retry
		return false, createErr
	})

	if err != nil {
		glog.Errorf(log("Attacher.inlineProvision inline volume provision failed %s: %v", volSpecName, err))
		return nil, err
	}

	glog.V(4).Info(log("Attacher.inlineProvision volume provisioned OK [Name: %s, ID:%s]", volSpecName, volume.Id))

	return volume, nil
}
