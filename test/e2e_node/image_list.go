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

	"k8s.io/apimachinery/pkg/util/sets"
	commontest "k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	// Number of attempts to pull an image.
	maxImagePullRetries = 5
	// Sleep duration between image pull retry attempts.
	imagePullRetryDelay = time.Second
)

// NodeImageWhiteList is a list of images used in node e2e test. These images will be prepulled
// before test running so that the image pulling won't fail in actual test.
var NodeImageWhiteList = sets.NewString(
	"google/cadvisor:latest",
	"gcr.io/google-containers/stress:v1",
	"gcr.io/google_containers/busybox:1.24",
	"gcr.io/google_containers/busybox@sha256:4bdd623e848417d96127e16037743f0cd8b528c026e9175e22a84f639eca58ff",
	"gcr.io/google_containers/node-problem-detector:v0.3.0",
	"gcr.io/google_containers/nginx-slim:0.7",
	"gcr.io/google_containers/serve_hostname:v1.4",
	"gcr.io/google_containers/netexec:1.7",
	framework.GetPauseImageNameForHostArch(),
)

func init() {
	// Union NodeImageWhiteList and CommonImageWhiteList into the framework image white list.
	framework.ImageWhiteList = NodeImageWhiteList.Union(commontest.CommonImageWhiteList)
}

// Pre-fetch all images tests depend on so that we don't fail in an actual test.
func PrePullAllImages() error {
	usr, err := user.Current()
	if err != nil {
		return err
	}
	images := framework.ImageWhiteList.List()
	glog.V(4).Infof("Pre-pulling images %+v", images)
	for _, image := range images {
		var (
			err    error
			output []byte
		)
		for i := 0; i < maxImagePullRetries; i++ {
			if i > 0 {
				time.Sleep(imagePullRetryDelay)
			}
			// TODO(random-liu): Use docker client to get rid of docker binary dependency.
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
