/*
Copyright 2016 The Kubernetes Authors.

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

package e2e_node

import (
	"os/exec"
	"os/user"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	// Number of attempts to pull an image.
	maxImagePullRetries = 5
	// Sleep duration between image pull retry attempts.
	imagePullRetryDelay = time.Second
	busyBoxImage        = iota

	hostExecImage
	netExecImage
	nginxImage
	mountTestImage
	testWebServer
	pauseImage

	// Images just used for explicitly testing pulling of images
	pullTestAlpine
	pullTestAlpineWithBash
	pullTestAuthenticatedAlpine
	pullTestExecHealthz
)

var ImageRegistry = map[int]string{
	busyBoxImage:   "gcr.io/google_containers/busybox:1.24",
	hostExecImage:  "gcr.io/google_containers/hostexec:1.2",
	netExecImage:   "gcr.io/google_containers/netexec:1.4",
	nginxImage:     "gcr.io/google_containers/nginx-slim:0.7",
	mountTestImage: "gcr.io/google_containers/mounttest:0.6",
	testWebServer:  "gcr.io/google_containers/test-webserver:e2e",
	pauseImage:     framework.GetPauseImageNameForHostArch(),
}

// These are used by tests that explicitly test the ability to pull images
var NoPullImageRegistry = map[int]string{
	pullTestExecHealthz:         "gcr.io/google_containers/exechealthz:1.0",
	pullTestAlpine:              "alpine:3.1",
	pullTestAlpineWithBash:      "gcr.io/google_containers/alpine-with-bash:1.0",
	pullTestAuthenticatedAlpine: "gcr.io/authenticated-image-pulling/alpine:3.1",
}

// Pre-fetch all images tests depend on so that we don't fail in an actual test
func PrePullAllImages() error {
	usr, err := user.Current()
	if err != nil {
		return err
	}
	for _, image := range ImageRegistry {
		var (
			err    error
			output []byte
		)
		for i := 0; i < maxImagePullRetries; i++ {
			if i > 0 {
				time.Sleep(imagePullRetryDelay)
			}
			if output, err = exec.Command("docker", "pull", image).CombinedOutput(); err == nil {
				break
			}
			glog.Warningf("Failed to pull %s as user %q, retrying in %s (%d of %d): %v",
				image, usr.Username, imagePullRetryDelay.String(), i+1, maxImagePullRetries, err)
		}
		if err != nil {
			glog.Warningf("Could not pre-pull image %s %v output:  %s", image, err, output)
			return err
		}
	}
	return nil
}
