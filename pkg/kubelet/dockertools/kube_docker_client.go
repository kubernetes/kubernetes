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

package dockertools

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"sync"
	"time"

	"github.com/golang/glog"

	dockermessage "github.com/docker/docker/pkg/jsonmessage"
	dockerstdcopy "github.com/docker/docker/pkg/stdcopy"
	dockerapi "github.com/docker/engine-api/client"
	dockertypes "github.com/docker/engine-api/types"
	"golang.org/x/net/context"
)

// kubeDockerClient is a wrapped layer of docker client for kubelet internal use. This layer is added to:
//	1) Redirect stream for exec and attach operations.
//	2) Wrap the context in this layer to make the DockerInterface cleaner.
//	3) Stabilize the DockerInterface. The engine-api is still under active development, the interface
//	is not stabilized yet. However, the DockerInterface is used in many files in Kubernetes, we may
//	not want to change the interface frequently. With this layer, we can port the engine api to the
//	DockerInterface to avoid changing DockerInterface as much as possible.
//	(See
//	  * https://github.com/docker/engine-api/issues/89
//	  * https://github.com/docker/engine-api/issues/137
//	  * https://github.com/docker/engine-api/pull/140)
// TODO(random-liu): Swith to new docker interface by refactoring the functions in the old DockerInterface
// one by one.
type kubeDockerClient struct {
	// timeout is the timeout of short running docker operations.
	timeout time.Duration
	client  *dockerapi.Client
}

// Make sure that kubeDockerClient implemented the DockerInterface.
var _ DockerInterface = &kubeDockerClient{}

// There are 2 kinds of docker operations categorized by running time:
// * Long running operation: The long running operation could run for arbitrary long time, and the running time
// usually depends on some uncontrollable factors. These operations include: PullImage, Logs, StartExec, AttachToContainer.
// * Non-long running operation: Given the maximum load of the system, the non-long running operation should finish
// in expected and usually short time. These include all other operations.
// kubeDockerClient only applies timeout on non-long running operations.
const (
	// defaultTimeout is the default timeout of short running docker operations.
	defaultTimeout = 2 * time.Minute

	// defaultShmSize is the default ShmSize to use (in bytes) if not specified.
	defaultShmSize = int64(1024 * 1024 * 64)

	// defaultImagePullingProgressReportInterval is the default interval of image pulling progress reporting.
	defaultImagePullingProgressReportInterval = 10 * time.Second

	// defaultImagePullingStuckTimeout is the default timeout for image pulling stuck. If no progress
	// is made for defaultImagePullingStuckTimeout, the image pulling will be cancelled.
	// Docker reports image progress for every 512kB block, so normally there shouldn't be too long interval
	// between progress updates.
	// TODO(random-liu): Make this configurable
	defaultImagePullingStuckTimeout = 1 * time.Minute
)

// newKubeDockerClient creates an kubeDockerClient from an existing docker client. If requestTimeout is 0,
// defaultTimeout will be applied.
func newKubeDockerClient(dockerClient *dockerapi.Client, requestTimeout time.Duration) DockerInterface {
	if requestTimeout == 0 {
		requestTimeout = defaultTimeout
	}

	k := &kubeDockerClient{
		client:  dockerClient,
		timeout: requestTimeout,
	}
	// Notice that this assumes that docker is running before kubelet is started.
	v, err := k.Version()
	if err != nil {
		glog.Errorf("failed to retrieve docker version: %v", err)
		glog.Warningf("Using empty version for docker client, this may sometimes cause compatibility issue.")
	} else {
		// Update client version with real api version.
		dockerClient.UpdateClientVersion(v.APIVersion)
	}
	return k
}

func (d *kubeDockerClient) ListContainers(options dockertypes.ContainerListOptions) ([]dockertypes.Container, error) {
	ctx, cancel := d.getTimeoutContext()
	defer cancel()
	containers, err := d.client.ContainerList(ctx, options)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		return nil, err
	}
	return containers, nil
}

func (d *kubeDockerClient) InspectContainer(id string) (*dockertypes.ContainerJSON, error) {
	ctx, cancel := d.getTimeoutContext()
	defer cancel()
	containerJSON, err := d.client.ContainerInspect(ctx, id)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		if dockerapi.IsErrContainerNotFound(err) {
			return nil, containerNotFoundError{ID: id}
		}
		return nil, err
	}
	return &containerJSON, nil
}

func (d *kubeDockerClient) CreateContainer(opts dockertypes.ContainerCreateConfig) (*dockertypes.ContainerCreateResponse, error) {
	ctx, cancel := d.getTimeoutContext()
	defer cancel()
	// we provide an explicit default shm size as to not depend on docker daemon.
	// TODO: evaluate exposing this as a knob in the API
	if opts.HostConfig != nil && opts.HostConfig.ShmSize <= 0 {
		opts.HostConfig.ShmSize = defaultShmSize
	}
	createResp, err := d.client.ContainerCreate(ctx, opts.Config, opts.HostConfig, opts.NetworkingConfig, opts.Name)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		return nil, err
	}
	return &createResp, nil
}

func (d *kubeDockerClient) StartContainer(id string) error {
	ctx, cancel := d.getTimeoutContext()
	defer cancel()
	err := d.client.ContainerStart(ctx, id)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return ctxErr
	}
	return err
}

// Stopping an already stopped container will not cause an error in engine-api.
func (d *kubeDockerClient) StopContainer(id string, timeout int) error {
	ctx, cancel := d.getCustomTimeoutContext(time.Duration(timeout) * time.Second)
	defer cancel()
	err := d.client.ContainerStop(ctx, id, timeout)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return ctxErr
	}
	return err
}

func (d *kubeDockerClient) RemoveContainer(id string, opts dockertypes.ContainerRemoveOptions) error {
	ctx, cancel := d.getTimeoutContext()
	defer cancel()
	err := d.client.ContainerRemove(ctx, id, opts)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return ctxErr
	}
	return err
}

func (d *kubeDockerClient) inspectImageRaw(ref string) (*dockertypes.ImageInspect, error) {
	ctx, cancel := d.getTimeoutContext()
	defer cancel()
	resp, _, err := d.client.ImageInspectWithRaw(ctx, ref, true)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		if dockerapi.IsErrImageNotFound(err) {
			err = imageNotFoundError{ID: ref}
		}
		return nil, err
	}

	return &resp, nil
}

func (d *kubeDockerClient) InspectImageByID(imageID string) (*dockertypes.ImageInspect, error) {
	resp, err := d.inspectImageRaw(imageID)
	if err != nil {
		return nil, err
	}

	if !matchImageIDOnly(*resp, imageID) {
		return nil, imageNotFoundError{ID: imageID}
	}
	return resp, nil
}

func (d *kubeDockerClient) InspectImageByRef(imageRef string) (*dockertypes.ImageInspect, error) {
	resp, err := d.inspectImageRaw(imageRef)
	if err != nil {
		return nil, err
	}

	if !matchImageTagOrSHA(*resp, imageRef) {
		return nil, imageNotFoundError{ID: imageRef}
	}
	return resp, nil
}

func (d *kubeDockerClient) ImageHistory(id string) ([]dockertypes.ImageHistory, error) {
	ctx, cancel := d.getTimeoutContext()
	defer cancel()
	resp, err := d.client.ImageHistory(ctx, id)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	return resp, err
}

func (d *kubeDockerClient) ListImages(opts dockertypes.ImageListOptions) ([]dockertypes.Image, error) {
	ctx, cancel := d.getTimeoutContext()
	defer cancel()
	images, err := d.client.ImageList(ctx, opts)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		return nil, err
	}
	return images, nil
}

func base64EncodeAuth(auth dockertypes.AuthConfig) (string, error) {
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(auth); err != nil {
		return "", err
	}
	return base64.URLEncoding.EncodeToString(buf.Bytes()), nil
}

// progress is a wrapper of dockermessage.JSONMessage with a lock protecting it.
type progress struct {
	sync.RWMutex
	// message stores the latest docker json message.
	message *dockermessage.JSONMessage
	// timestamp of the latest update.
	timestamp time.Time
}

func newProgress() *progress {
	return &progress{timestamp: time.Now()}
}

func (p *progress) set(msg *dockermessage.JSONMessage) {
	p.Lock()
	defer p.Unlock()
	p.message = msg
	p.timestamp = time.Now()
}

func (p *progress) get() (string, time.Time) {
	p.RLock()
	defer p.RUnlock()
	if p.message == nil {
		return "No progress", p.timestamp
	}
	// The following code is based on JSONMessage.Display
	var prefix string
	if p.message.ID != "" {
		prefix = fmt.Sprintf("%s: ", p.message.ID)
	}
	if p.message.Progress == nil {
		return fmt.Sprintf("%s%s", prefix, p.message.Status), p.timestamp
	}
	return fmt.Sprintf("%s%s %s", prefix, p.message.Status, p.message.Progress.String()), p.timestamp
}

// progressReporter keeps the newest image pulling progress and periodically report the newest progress.
type progressReporter struct {
	*progress
	image  string
	cancel context.CancelFunc
	stopCh chan struct{}
}

// newProgressReporter creates a new progressReporter for specific image with specified reporting interval
func newProgressReporter(image string, cancel context.CancelFunc) *progressReporter {
	return &progressReporter{
		progress: newProgress(),
		image:    image,
		cancel:   cancel,
		stopCh:   make(chan struct{}),
	}
}

// start starts the progressReporter
func (p *progressReporter) start() {
	go func() {
		ticker := time.NewTicker(defaultImagePullingProgressReportInterval)
		defer ticker.Stop()
		for {
			// TODO(random-liu): Report as events.
			select {
			case <-ticker.C:
				progress, timestamp := p.progress.get()
				// If there is no progress for defaultImagePullingStuckTimeout, cancel the operation.
				if time.Now().Sub(timestamp) > defaultImagePullingStuckTimeout {
					glog.Errorf("Cancel pulling image %q because of no progress for %v, latest progress: %q", p.image, defaultImagePullingStuckTimeout, progress)
					p.cancel()
					return
				}
				glog.V(2).Infof("Pulling image %q: %q", p.image, progress)
			case <-p.stopCh:
				progress, _ := p.progress.get()
				glog.V(2).Infof("Stop pulling image %q: %q", p.image, progress)
				return
			}
		}
	}()
}

// stop stops the progressReporter
func (p *progressReporter) stop() {
	close(p.stopCh)
}

func (d *kubeDockerClient) PullImage(image string, auth dockertypes.AuthConfig, opts dockertypes.ImagePullOptions) error {
	// RegistryAuth is the base64 encoded credentials for the registry
	base64Auth, err := base64EncodeAuth(auth)
	if err != nil {
		return err
	}
	opts.RegistryAuth = base64Auth
	ctx, cancel := d.getCancelableContext()
	defer cancel()
	resp, err := d.client.ImagePull(ctx, image, opts)
	if err != nil {
		return err
	}
	defer resp.Close()
	reporter := newProgressReporter(image, cancel)
	reporter.start()
	defer reporter.stop()
	decoder := json.NewDecoder(resp)
	for {
		var msg dockermessage.JSONMessage
		err := decoder.Decode(&msg)
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if msg.Error != nil {
			return msg.Error
		}
		reporter.set(&msg)
	}
	return nil
}

func (d *kubeDockerClient) RemoveImage(image string, opts dockertypes.ImageRemoveOptions) ([]dockertypes.ImageDelete, error) {
	ctx, cancel := d.getTimeoutContext()
	defer cancel()
	resp, err := d.client.ImageRemove(ctx, image, opts)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	return resp, err
}

func (d *kubeDockerClient) Logs(id string, opts dockertypes.ContainerLogsOptions, sopts StreamOptions) error {
	ctx, cancel := d.getCancelableContext()
	defer cancel()
	resp, err := d.client.ContainerLogs(ctx, id, opts)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return ctxErr
	}
	if err != nil {
		return err
	}
	defer resp.Close()
	return d.redirectResponseToOutputStream(sopts.RawTerminal, sopts.OutputStream, sopts.ErrorStream, resp)
}

func (d *kubeDockerClient) Version() (*dockertypes.Version, error) {
	ctx, cancel := d.getTimeoutContext()
	defer cancel()
	resp, err := d.client.ServerVersion(ctx)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

func (d *kubeDockerClient) Info() (*dockertypes.Info, error) {
	ctx, cancel := d.getTimeoutContext()
	defer cancel()
	resp, err := d.client.Info(ctx)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

// TODO(random-liu): Add unit test for exec and attach functions, just like what go-dockerclient did.
func (d *kubeDockerClient) CreateExec(id string, opts dockertypes.ExecConfig) (*dockertypes.ContainerExecCreateResponse, error) {
	ctx, cancel := d.getTimeoutContext()
	defer cancel()
	resp, err := d.client.ContainerExecCreate(ctx, id, opts)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

func (d *kubeDockerClient) StartExec(startExec string, opts dockertypes.ExecStartCheck, sopts StreamOptions) error {
	ctx, cancel := d.getCancelableContext()
	defer cancel()
	if opts.Detach {
		err := d.client.ContainerExecStart(ctx, startExec, opts)
		if ctxErr := contextError(ctx); ctxErr != nil {
			return ctxErr
		}
		return err
	}
	resp, err := d.client.ContainerExecAttach(ctx, startExec, dockertypes.ExecConfig{
		Detach: opts.Detach,
		Tty:    opts.Tty,
	})
	if ctxErr := contextError(ctx); ctxErr != nil {
		return ctxErr
	}
	if err != nil {
		return err
	}
	defer resp.Close()
	return d.holdHijackedConnection(sopts.RawTerminal || opts.Tty, sopts.InputStream, sopts.OutputStream, sopts.ErrorStream, resp)
}

func (d *kubeDockerClient) InspectExec(id string) (*dockertypes.ContainerExecInspect, error) {
	ctx, cancel := d.getTimeoutContext()
	defer cancel()
	resp, err := d.client.ContainerExecInspect(ctx, id)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

func (d *kubeDockerClient) AttachToContainer(id string, opts dockertypes.ContainerAttachOptions, sopts StreamOptions) error {
	ctx, cancel := d.getCancelableContext()
	defer cancel()
	resp, err := d.client.ContainerAttach(ctx, id, opts)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return ctxErr
	}
	if err != nil {
		return err
	}
	defer resp.Close()
	return d.holdHijackedConnection(sopts.RawTerminal, sopts.InputStream, sopts.OutputStream, sopts.ErrorStream, resp)
}

func (d *kubeDockerClient) ResizeExecTTY(id string, height, width int) error {
	ctx, cancel := d.getCancelableContext()
	defer cancel()
	return d.client.ContainerExecResize(ctx, id, dockertypes.ResizeOptions{
		Height: height,
		Width:  width,
	})
}

func (d *kubeDockerClient) ResizeContainerTTY(id string, height, width int) error {
	ctx, cancel := d.getCancelableContext()
	defer cancel()
	return d.client.ContainerResize(ctx, id, dockertypes.ResizeOptions{
		Height: height,
		Width:  width,
	})
}

// redirectResponseToOutputStream redirect the response stream to stdout and stderr. When tty is true, all stream will
// only be redirected to stdout.
func (d *kubeDockerClient) redirectResponseToOutputStream(tty bool, outputStream, errorStream io.Writer, resp io.Reader) error {
	if outputStream == nil {
		outputStream = ioutil.Discard
	}
	if errorStream == nil {
		errorStream = ioutil.Discard
	}
	var err error
	if tty {
		_, err = io.Copy(outputStream, resp)
	} else {
		_, err = dockerstdcopy.StdCopy(outputStream, errorStream, resp)
	}
	return err
}

// holdHijackedConnection hold the HijackedResponse, redirect the inputStream to the connection, and redirect the response
// stream to stdout and stderr. NOTE: If needed, we could also add context in this function.
func (d *kubeDockerClient) holdHijackedConnection(tty bool, inputStream io.Reader, outputStream, errorStream io.Writer, resp dockertypes.HijackedResponse) error {
	receiveStdout := make(chan error)
	if outputStream != nil || errorStream != nil {
		go func() {
			receiveStdout <- d.redirectResponseToOutputStream(tty, outputStream, errorStream, resp.Reader)
		}()
	}

	stdinDone := make(chan struct{})
	go func() {
		if inputStream != nil {
			io.Copy(resp.Conn, inputStream)
		}
		resp.CloseWrite()
		close(stdinDone)
	}()

	select {
	case err := <-receiveStdout:
		return err
	case <-stdinDone:
		if outputStream != nil || errorStream != nil {
			return <-receiveStdout
		}
	}
	return nil
}

// getCancelableContext returns a new cancelable context. For long running requests without timeout, we use cancelable
// context to avoid potential resource leak, although the current implementation shouldn't leak resource.
func (d *kubeDockerClient) getCancelableContext() (context.Context, context.CancelFunc) {
	return context.WithCancel(context.Background())
}

// getTimeoutContext returns a new context with default request timeout
func (d *kubeDockerClient) getTimeoutContext() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), d.timeout)
}

// getCustomTimeoutContext returns a new context with a specific request timeout
func (d *kubeDockerClient) getCustomTimeoutContext(timeout time.Duration) (context.Context, context.CancelFunc) {
	// Pick the larger of the two
	if d.timeout > timeout {
		timeout = d.timeout
	}
	return context.WithTimeout(context.Background(), timeout)
}

// ParseDockerTimestamp parses the timestamp returned by DockerInterface from string to time.Time
func ParseDockerTimestamp(s string) (time.Time, error) {
	// Timestamp returned by Docker is in time.RFC3339Nano format.
	return time.Parse(time.RFC3339Nano, s)
}

// contextError checks the context, and returns error if the context is timeout.
func contextError(ctx context.Context) error {
	if ctx.Err() == context.DeadlineExceeded {
		return operationTimeout{err: ctx.Err()}
	}
	return ctx.Err()
}

// StreamOptions are the options used to configure the stream redirection
type StreamOptions struct {
	RawTerminal  bool
	InputStream  io.Reader
	OutputStream io.Writer
	ErrorStream  io.Writer
}

// operationTimeout is the error returned when the docker operations are timeout.
type operationTimeout struct {
	err error
}

func (e operationTimeout) Error() string {
	return fmt.Sprintf("operation timeout: %v", e.err)
}

// containerNotFoundError is the error returned by InspectContainer when container not found. We
// add this error type for testability. We don't use the original error returned by engine-api
// because dockertypes.containerNotFoundError is private, we can't create and inject it in our test.
type containerNotFoundError struct {
	ID string
}

func (e containerNotFoundError) Error() string {
	return fmt.Sprintf("no such container: %q", e.ID)
}

// imageNotFoundError is the error returned by InspectImage when image not found.
type imageNotFoundError struct {
	ID string
}

func (e imageNotFoundError) Error() string {
	return fmt.Sprintf("no such image: %q", e.ID)
}

// IsImageNotFoundError checks whether the error is image not found error. This is exposed
// to share with dockershim.
func IsImageNotFoundError(err error) bool {
	_, ok := err.(imageNotFoundError)
	return ok
}
