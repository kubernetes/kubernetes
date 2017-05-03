/*
Copyright 2014 The Kubernetes Authors.

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

package dockertools

import (
	"fmt"
	"net/http"
	"path"
	"strings"

	"github.com/docker/docker/pkg/jsonmessage"
	dockertypes "github.com/docker/engine-api/types"
	"github.com/golang/glog"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
	"k8s.io/kubernetes/pkg/kubelet/images"
)

const (
	LogSuffix          = "log"
	ext4MaxFileNameLen = 255

	DockerType = "docker"
)

// DockerPuller is an abstract interface for testability.  It abstracts image pull operations.
// DockerPuller is *not* in use anywhere in the codebase.
// TODO: Examine whether we can migrate the unit tests and remove the code.
type DockerPuller interface {
	Pull(image string, secrets []v1.Secret) error
	GetImageRef(image string) (string, error)
}

// dockerPuller is the default implementation of DockerPuller.
type dockerPuller struct {
	client  libdocker.Interface
	keyring credentialprovider.DockerKeyring
}

// newDockerPuller creates a new instance of the default implementation of DockerPuller.
func newDockerPuller(client libdocker.Interface) DockerPuller {
	return &dockerPuller{
		client:  client,
		keyring: credentialprovider.NewDockerKeyring(),
	}
}

func filterHTTPError(err error, image string) error {
	// docker/docker/pull/11314 prints detailed error info for docker pull.
	// When it hits 502, it returns a verbose html output including an inline svg,
	// which makes the output of kubectl get pods much harder to parse.
	// Here converts such verbose output to a concise one.
	jerr, ok := err.(*jsonmessage.JSONError)
	if ok && (jerr.Code == http.StatusBadGateway ||
		jerr.Code == http.StatusServiceUnavailable ||
		jerr.Code == http.StatusGatewayTimeout) {
		glog.V(2).Infof("Pulling image %q failed: %v", image, err)
		return images.RegistryUnavailable
	} else {
		return err
	}
}

func (p dockerPuller) Pull(image string, secrets []v1.Secret) error {
	keyring, err := credentialprovider.MakeDockerKeyring(secrets, p.keyring)
	if err != nil {
		return err
	}

	// The only used image pull option RegistryAuth will be set in kube_docker_client
	opts := dockertypes.ImagePullOptions{}

	creds, haveCredentials := keyring.Lookup(image)
	if !haveCredentials {
		glog.V(1).Infof("Pulling image %s without credentials", image)

		err := p.client.PullImage(image, dockertypes.AuthConfig{}, opts)
		if err == nil {
			// Sometimes PullImage failed with no error returned.
			imageRef, ierr := p.GetImageRef(image)
			if ierr != nil {
				glog.Warningf("Failed to inspect image %s: %v", image, ierr)
			}
			if imageRef == "" {
				return fmt.Errorf("image pull failed for unknown error")
			}
			return nil
		}

		// Image spec: [<registry>/]<repository>/<image>[:<version] so we count '/'
		explicitRegistry := (strings.Count(image, "/") == 2)
		// Hack, look for a private registry, and decorate the error with the lack of
		// credentials.  This is heuristic, and really probably could be done better
		// by talking to the registry API directly from the kubelet here.
		if explicitRegistry {
			return fmt.Errorf("image pull failed for %s, this may be because there are no credentials on this request.  details: (%v)", image, err)
		}

		return filterHTTPError(err, image)
	}

	var pullErrs []error
	for _, currentCreds := range creds {
		err = p.client.PullImage(image, credentialprovider.LazyProvide(currentCreds), opts)
		// If there was no error, return success
		if err == nil {
			return nil
		}

		pullErrs = append(pullErrs, filterHTTPError(err, image))
	}

	return utilerrors.NewAggregate(pullErrs)
}

func (p dockerPuller) GetImageRef(image string) (string, error) {
	resp, err := p.client.InspectImageByRef(image)
	if err == nil {
		if resp == nil {
			return "", nil
		}

		imageRef := resp.ID
		if len(resp.RepoDigests) > 0 {
			imageRef = resp.RepoDigests[0]
		}
		return imageRef, nil
	}
	if libdocker.IsImageNotFoundError(err) {
		return "", nil
	}
	return "", err
}

func LogSymlink(containerLogsDir, podFullName, containerName, dockerId string) string {
	suffix := fmt.Sprintf(".%s", LogSuffix)
	logPath := fmt.Sprintf("%s_%s-%s", podFullName, containerName, dockerId)
	// Length of a filename cannot exceed 255 characters in ext4 on Linux.
	if len(logPath) > ext4MaxFileNameLen-len(suffix) {
		logPath = logPath[:ext4MaxFileNameLen-len(suffix)]
	}
	return path.Join(containerLogsDir, logPath+suffix)
}
