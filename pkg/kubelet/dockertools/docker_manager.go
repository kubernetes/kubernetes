/*
Copyright 2015 The Kubernetes Authors.

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
	"bytes"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	dockertypes "github.com/docker/engine-api/types"
	"github.com/golang/glog"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/unversioned/remotecommand"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/security/apparmor"
)

const (
	DockerType                 = "docker"
	dockerDefaultLoggingDriver = "json-file"

	// Docker changed the API for specifying options in v1.11
	SecurityOptSeparatorChangeVersion = "1.23.0" // Corresponds to docker 1.11.x
	SecurityOptSeparatorOld           = ':'
	SecurityOptSeparatorNew           = '='

	// https://docs.docker.com/engine/reference/api/docker_remote_api/
	// docker version should be at least 1.10.x
	minimumDockerAPIVersion = "1.22"

	statusRunningPrefix = "Up"
	statusExitedPrefix  = "Exited"
	statusCreatedPrefix = "Created"

	ndotsDNSOption = "options ndots:5\n"
)

var (
	defaultSeccompOpt = []dockerOpt{{"seccomp", "unconfined", ""}}
)

// GetImageRef returns the image digest if exists, or else returns the image ID.
// It is exported for reusing in dockershim.
func GetImageRef(client DockerInterface, image string) (string, error) {
	img, err := client.InspectImageByRef(image)
	if err != nil {
		return "", err
	}
	if img == nil {
		return "", fmt.Errorf("unable to inspect image %s", image)
	}

	// Returns the digest if it exist.
	if len(img.RepoDigests) > 0 {
		return img.RepoDigests[0], nil
	}

	return img.ID, nil
}

// Temporarily export this function to share with dockershim.
// TODO: clean this up.
func GetContainerLogs(client DockerInterface, pod *v1.Pod, containerID kubecontainer.ContainerID, logOptions *v1.PodLogOptions, stdout, stderr io.Writer, rawTerm bool) error {
	var since int64
	if logOptions.SinceSeconds != nil {
		t := metav1.Now().Add(-time.Duration(*logOptions.SinceSeconds) * time.Second)
		since = t.Unix()
	}
	if logOptions.SinceTime != nil {
		since = logOptions.SinceTime.Unix()
	}
	opts := dockertypes.ContainerLogsOptions{
		ShowStdout: true,
		ShowStderr: true,
		Since:      strconv.FormatInt(since, 10),
		Timestamps: logOptions.Timestamps,
		Follow:     logOptions.Follow,
	}
	if logOptions.TailLines != nil {
		opts.Tail = strconv.FormatInt(*logOptions.TailLines, 10)
	}

	sopts := StreamOptions{
		OutputStream: stdout,
		ErrorStream:  stderr,
		RawTerminal:  rawTerm,
	}
	return client.Logs(containerID.ID, opts, sopts)
}

// Temporarily export this function to share with dockershim.
// TODO: clean this up.
func AttachContainer(client DockerInterface, containerID string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize) error {
	// Have to start this before the call to client.AttachToContainer because client.AttachToContainer is a blocking
	// call :-( Otherwise, resize events don't get processed and the terminal never resizes.
	kubecontainer.HandleResizing(resize, func(size remotecommand.TerminalSize) {
		client.ResizeContainerTTY(containerID, int(size.Height), int(size.Width))
	})

	// TODO(random-liu): Do we really use the *Logs* field here?
	opts := dockertypes.ContainerAttachOptions{
		Stream: true,
		Stdin:  stdin != nil,
		Stdout: stdout != nil,
		Stderr: stderr != nil,
	}
	sopts := StreamOptions{
		InputStream:  stdin,
		OutputStream: stdout,
		ErrorStream:  stderr,
		RawTerminal:  tty,
	}
	return client.AttachToContainer(containerID, opts, sopts)
}

// Temporarily export this function to share with dockershim.
func PortForward(client DockerInterface, podInfraContainerID string, port int32, stream io.ReadWriteCloser) error {
	container, err := client.InspectContainer(podInfraContainerID)
	if err != nil {
		return err
	}

	if !container.State.Running {
		return fmt.Errorf("container not running (%s)", container.ID)
	}

	containerPid := container.State.Pid
	socatPath, lookupErr := exec.LookPath("socat")
	if lookupErr != nil {
		return fmt.Errorf("unable to do port forwarding: socat not found.")
	}

	args := []string{"-t", fmt.Sprintf("%d", containerPid), "-n", socatPath, "-", fmt.Sprintf("TCP4:localhost:%d", port)}

	nsenterPath, lookupErr := exec.LookPath("nsenter")
	if lookupErr != nil {
		return fmt.Errorf("unable to do port forwarding: nsenter not found.")
	}

	commandString := fmt.Sprintf("%s %s", nsenterPath, strings.Join(args, " "))
	glog.V(4).Infof("executing port forwarding command: %s", commandString)

	command := exec.Command(nsenterPath, args...)
	command.Stdout = stream

	stderr := new(bytes.Buffer)
	command.Stderr = stderr

	// If we use Stdin, command.Run() won't return until the goroutine that's copying
	// from stream finishes. Unfortunately, if you have a client like telnet connected
	// via port forwarding, as long as the user's telnet client is connected to the user's
	// local listener that port forwarding sets up, the telnet session never exits. This
	// means that even if socat has finished running, command.Run() won't ever return
	// (because the client still has the connection and stream open).
	//
	// The work around is to use StdinPipe(), as Wait() (called by Run()) closes the pipe
	// when the command (socat) exits.
	inPipe, err := command.StdinPipe()
	if err != nil {
		return fmt.Errorf("unable to do port forwarding: error creating stdin pipe: %v", err)
	}
	go func() {
		io.Copy(inPipe, stream)
		inPipe.Close()
	}()

	if err := command.Run(); err != nil {
		return fmt.Errorf("%v: %s", err, stderr.String())
	}

	return nil
}

// Temporarily export this function to share with dockershim.
// TODO: clean this up.
func GetAppArmorOpts(profile string) ([]dockerOpt, error) {
	if profile == "" || profile == apparmor.ProfileRuntimeDefault {
		// The docker applies the default profile by default.
		return nil, nil
	}

	// Assume validation has already happened.
	profileName := strings.TrimPrefix(profile, apparmor.ProfileNamePrefix)
	return []dockerOpt{{"apparmor", profileName, ""}}, nil
}

// Temporarily export this function to share with dockershim.
// TODO: clean this up.
func GetSeccompOpts(annotations map[string]string, ctrName, profileRoot string) ([]dockerOpt, error) {
	profile, profileOK := annotations[v1.SeccompContainerAnnotationKeyPrefix+ctrName]
	if !profileOK {
		// try the pod profile
		profile, profileOK = annotations[v1.SeccompPodAnnotationKey]
		if !profileOK {
			// return early the default
			return defaultSeccompOpt, nil
		}
	}

	if profile == "unconfined" {
		// return early the default
		return defaultSeccompOpt, nil
	}

	if profile == "docker/default" {
		// return nil so docker will load the default seccomp profile
		return nil, nil
	}

	if !strings.HasPrefix(profile, "localhost/") {
		return nil, fmt.Errorf("unknown seccomp profile option: %s", profile)
	}

	name := strings.TrimPrefix(profile, "localhost/") // by pod annotation validation, name is a valid subpath
	fname := filepath.Join(profileRoot, filepath.FromSlash(name))
	file, err := ioutil.ReadFile(fname)
	if err != nil {
		return nil, fmt.Errorf("cannot load seccomp profile %q: %v", name, err)
	}

	b := bytes.NewBuffer(nil)
	if err := json.Compact(b, file); err != nil {
		return nil, err
	}
	// Rather than the full profile, just put the filename & md5sum in the event log.
	msg := fmt.Sprintf("%s(md5:%x)", name, md5.Sum(file))

	return []dockerOpt{{"seccomp", b.String(), msg}}, nil
}

// FmtDockerOpts formats the docker security options using the given separator.
func FmtDockerOpts(opts []dockerOpt, sep rune) []string {
	fmtOpts := make([]string, len(opts))
	for i, opt := range opts {
		fmtOpts[i] = fmt.Sprintf("%s%c%s", opt.key, sep, opt.value)
	}
	return fmtOpts
}

type dockerOpt struct {
	// The key-value pair passed to docker.
	key, value string
	// The alternative value to use in log/event messages.
	msg string
}

// Expose key/value from dockertools
func (d dockerOpt) GetKV() (string, string) {
	return d.key, d.value
}

// GetUserFromImageUser splits the user out of an user:group string.
func GetUserFromImageUser(id string) string {
	if id == "" {
		return id
	}
	// split instances where the id may contain user:group
	if strings.Contains(id, ":") {
		return strings.Split(id, ":")[0]
	}
	// no group, just return the id
	return id
}

// RewriteResolvFile rewrites resolv.conf file generated by docker.
// Exported for reusing in dockershim.
func RewriteResolvFile(resolvFilePath string, dns []string, dnsSearch []string, useClusterFirstPolicy bool) error {
	if len(resolvFilePath) == 0 {
		glog.Errorf("ResolvConfPath is empty.")
		return nil
	}

	if _, err := os.Stat(resolvFilePath); os.IsNotExist(err) {
		return fmt.Errorf("ResolvConfPath %q does not exist", resolvFilePath)
	}

	var resolvFileContent []string

	for _, srv := range dns {
		resolvFileContent = append(resolvFileContent, "nameserver "+srv)
	}

	if len(dnsSearch) > 0 {
		resolvFileContent = append(resolvFileContent, "search "+strings.Join(dnsSearch, " "))
	}

	if len(resolvFileContent) > 0 {
		if useClusterFirstPolicy {
			resolvFileContent = append(resolvFileContent, ndotsDNSOption)
		}

		resolvFileContentStr := strings.Join(resolvFileContent, "\n")
		resolvFileContentStr += "\n"

		glog.V(4).Infof("Will attempt to re-write config file %s with: \n%s", resolvFilePath, resolvFileContent)
		if err := rewriteFile(resolvFilePath, resolvFileContentStr); err != nil {
			glog.Errorf("resolv.conf could not be updated: %v", err)
			return err
		}
	}

	return nil
}

func rewriteFile(filePath, stringToWrite string) error {
	f, err := os.OpenFile(filePath, os.O_TRUNC|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = f.WriteString(stringToWrite)
	return err
}
