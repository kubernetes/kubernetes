// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package integration

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
)

func buildDockerImage(imageName string) error {
	out, err := exec.Command("./build.sh", imageName).CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to build docker binary (%q) - %q", err, out)
	}

	return nil
}

func copyDockerImage(imageName, hostname, zone string) error {
	tempfile, err := ioutil.TempFile("", hostname)
	if err != nil {
		return err
	}
	defer os.Remove(tempfile.Name())
	out, err := exec.Command("docker", "save", "-o", tempfile.Name(), imageName).CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to save docker binary (%q) - %q", err, out)
	}
	remoteFile := path.Join("/tmp", path.Base(tempfile.Name()))
	out, err = exec.Command("gcloud", "compute", "copy-files", "--zone", zone, tempfile.Name(), fmt.Sprintf("%s:%s", hostname, remoteFile)).CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to push docker binary to %q (%q) - %q", hostname, err, out)
	}
	out, err = exec.Command("gcloud", "compute", "ssh", "--zone", zone, hostname, "--command", fmt.Sprintf("sudo docker load -i %s", remoteFile)).CombinedOutput()
	if err != nil {
		err = fmt.Errorf("failed to load docker image %q using temp file %q on host %q (%q) - %q", imageName, remoteFile, hostname, err, out)
	}
	out, rmErr := exec.Command("gcloud", "compute", "ssh", "--zone", zone, hostname, "--command", fmt.Sprintf("sudo rm -f %s", remoteFile)).CombinedOutput()
	if rmErr != nil {
		if err != nil {
			rmErr = fmt.Errorf("%v\nfailed to remove tempfile on host %q (%q) - %q", err, hostname, err, out)
		}
		return rmErr
	}
	return err
}

func removeDockerImage(imageName string) error {
	out, err := exec.Command("docker", "rmi", "-f", imageName).CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to remove docker image %q (%q) - %q", imageName, err, out)
	}
	return nil
}

func cleanupRemoteHost(hostname, zone string) {
	_ = exec.Command("gcloud", "compute", "ssh", "--zone", zone, hostname, "--command", "\"sudo docker rm `docker ps -a -q`\"")
	_ = exec.Command("gcloud", "compute", "ssh", "--zone", zone, hostname, "--command", "\"sudo docker rmi `docker images -a -q`\"")
}
