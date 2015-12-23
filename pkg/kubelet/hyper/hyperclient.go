/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package hyper

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"mime"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"strings"

	"github.com/docker/docker/pkg/parsers"
	"github.com/docker/docker/pkg/stdcopy"
	"github.com/golang/glog"
	"time"
)

const (
	HYPER_PROTO       = "unix"
	HYPER_ADDR        = "/var/run/hyper.sock"
	HYPER_SCHEME      = "http"
	HYPER_MINVERSION  = "0.3.0"
	DEFAULT_IMAGE_TAG = "latest"

	KEY_ID             = "id"
	KEY_IMAGEID        = "imageId"
	KEY_IMAGENAME      = "imageName"
	KEY_ITEM           = "item"
	KEY_DNS            = "dns"
	KEY_MEMORY         = "memory"
	KEY_POD_ARGS       = "podArgs"
	KEY_POD_ID         = "podId"
	KEY_POD_NAME       = "podName"
	KEY_RESOURCE       = "resource"
	KEY_VCPU           = "vcpu"
	KEY_TTY            = "tty"
	KEY_TYPE           = "type"
	KEY_VALUE          = "value"
	KEY_NAME           = "name"
	KEY_IMAGE          = "image"
	KEY_VOLUMES        = "volumes"
	KEY_CONTAINERS     = "containers"
	KEY_VOLUME_SOURCE  = "source"
	KEY_VOLUME_DRIVE   = "driver"
	KEY_ENVS           = "envs"
	KEY_CONTAINER_PORT = "containerPort"
	KEY_HOST_PORT      = "hostPort"
	KEY_PROTOCOL       = "protocol"
	KEY_PORTS          = "ports"
	KEY_MOUNTPATH      = "path"
	KEY_READONLY       = "readOnly"
	KEY_VOLUME         = "volume"
	KEY_COMMAND        = "command"
	KEY_CONTAINER_ARGS = "args"
	KEY_WORKDIR        = "workdir"
	VOLUME_TYPE_VFS    = "vfs"
	TYPE_CONTAINER     = "container"
	TYPE_POD           = "pod"
)

type HyperClient struct {
	proto  string
	addr   string
	scheme string
}

type AttachToContainerOptions struct {
	Container    string
	InputStream  io.Reader
	OutputStream io.Writer
	ErrorStream  io.Writer

	// Get container logs, sending it to OutputStream.
	Logs bool

	// Stream the response?
	Stream bool

	// Attach to stdin, and use InputStream.
	Stdin bool

	// Attach to stdout, and use OutputStream.
	Stdout bool

	// Attach to stderr, and use ErrorStream.
	Stderr bool

	// If set, after a successful connect, a sentinel will be sent and then the
	// client will block on receive before continuing.
	//
	// It must be an unbuffered channel. Using a buffered channel can lead
	// to unexpected behavior.
	Success chan struct{}

	// Use raw terminal? Usually true when the container contains a TTY.
	RawTerminal bool `qs:"-"`
}

type hijackOptions struct {
	success        chan struct{}
	setRawTerminal bool
	in             io.Reader
	stdout         io.Writer
	stderr         io.Writer
	data           interface{}
}

func NewHyperClient() *HyperClient {
	var (
		scheme = HYPER_SCHEME
		proto  = HYPER_PROTO
		addr   = HYPER_ADDR
	)

	return &HyperClient{
		proto:  proto,
		addr:   addr,
		scheme: scheme,
	}
}

var (
	ErrConnectionRefused = errors.New("Cannot connect to the Hyper daemon. Is 'hyperd' running on this host?")
)

func (cli *HyperClient) encodeData(data interface{}) (*bytes.Buffer, error) {
	params := bytes.NewBuffer(nil)
	if data != nil {
		buf, err := json.Marshal(data)
		if err != nil {
			return nil, err
		}
		if _, err := params.Write(buf); err != nil {
			return nil, err
		}
	}
	return params, nil
}

// parseImageName parses a docker image string into two parts: repo and tag.
// If tag is empty, return the defaultImageTag.
func parseImageName(image string) (string, string) {
	repoToPull, tag := parsers.ParseRepositoryTag(image)
	// If no tag was specified, use the default "latest".
	if len(tag) == 0 {
		tag = DEFAULT_IMAGE_TAG
	}
	return repoToPull, tag
}

func (cli *HyperClient) clientRequest(method, path string, in io.Reader, headers map[string][]string) (io.ReadCloser, string, int, error) {
	expectedPayload := (method == "POST" || method == "PUT")
	if expectedPayload && in == nil {
		in = bytes.NewReader([]byte{})
	}
	req, err := http.NewRequest(method, path, in)
	if err != nil {
		return nil, "", -1, err
	}
	req.Header.Set("User-Agent", "kubelet")
	req.URL.Host = cli.addr
	req.URL.Scheme = cli.scheme

	if headers != nil {
		for k, v := range headers {
			req.Header[k] = v
		}
	}

	if expectedPayload && req.Header.Get("Content-Type") == "" {
		req.Header.Set("Content-Type", "text/plain")
	}

	var dial net.Conn
	dial, err = net.DialTimeout(HYPER_PROTO, HYPER_ADDR, 32*time.Second)
	if err != nil {
		return nil, "", -1, err
	}
	defer dial.Close()

	clientconn := httputil.NewClientConn(dial, nil)
	defer clientconn.Close()

	resp, err := clientconn.Do(req)
	statusCode := -1
	if resp != nil {
		statusCode = resp.StatusCode
	}
	if err != nil {
		if strings.Contains(err.Error(), "connection refused") {
			return nil, "", statusCode, ErrConnectionRefused
		}

		return nil, "", statusCode, fmt.Errorf("An error occurred trying to connect: %v", err)
	}

	if statusCode < 200 || statusCode >= 400 {
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return nil, "", statusCode, err
		}
		if len(body) == 0 {
			return nil, "", statusCode, fmt.Errorf("Error: request returned %s for API route and version %s, check if the server supports the requested API version", http.StatusText(statusCode), req.URL)
		}
		if len(bytes.TrimSpace(body)) > 150 {
			return nil, "", statusCode, fmt.Errorf("Error from daemon's response")
		}
		return nil, "", statusCode, fmt.Errorf("%s", bytes.TrimSpace(body))
	}

	return resp.Body, resp.Header.Get("Content-Type"), statusCode, nil
}

func (cli *HyperClient) call(method, path string, data interface{}, headers map[string][]string) (io.ReadCloser, int, error) {
	params, err := cli.encodeData(data)
	if err != nil {
		return nil, -1, err
	}

	if data != nil {
		if headers == nil {
			headers = make(map[string][]string)
		}
		headers["Content-Type"] = []string{"application/json"}
	}

	body, _, statusCode, err := cli.clientRequest(method, path, params, headers)
	return body, statusCode, err
}

func (cli *HyperClient) stream(method, path string, in io.Reader, out io.Writer, headers map[string][]string) error {
	return cli.streamHelper(method, path, true, in, out, nil, headers)
}

func (cli *HyperClient) streamHelper(method, path string, setRawTerminal bool, in io.Reader, stdout, stderr io.Writer, headers map[string][]string) error {
	body, contentType, _, err := cli.clientRequest(method, path, in, headers)
	if err != nil {
		return err
	}
	return cli.streamBody(body, contentType, setRawTerminal, stdout, stderr)
}

func MatchesContentType(contentType, expectedType string) bool {
	mimetype, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		glog.V(4).Infof("Error parsing media type: %s error: %v", contentType, err)
	}
	return err == nil && mimetype == expectedType
}

func (cli *HyperClient) streamBody(body io.ReadCloser, contentType string, setRawTerminal bool, stdout, stderr io.Writer) error {
	defer body.Close()

	if MatchesContentType(contentType, "application/json") {
		buf := new(bytes.Buffer)
		buf.ReadFrom(body)
		if stdout != nil {
			stdout.Write(buf.Bytes())
		}
		return nil
	}
	return nil
}

func readBody(stream io.ReadCloser, statusCode int, err error) ([]byte, int, error) {
	if stream != nil {
		defer stream.Close()
	}
	if err != nil {
		return nil, statusCode, err
	}
	if stream == nil {
		return nil, statusCode, err
	}
	body, err := ioutil.ReadAll(stream)
	if err != nil {
		return nil, -1, err
	}
	return body, statusCode, nil
}

func (client *HyperClient) Version() (string, error) {
	body, _, err := readBody(client.call("GET", "/version", nil, nil))
	if err != nil {
		return "", err
	}

	var info map[string]interface{}
	err = json.Unmarshal(body, &info)
	if err != nil {
		return "", err
	}

	version, ok := info["Version"]
	if !ok {
		return "", fmt.Errorf("Can not get hyper version")
	}

	return version.(string), nil
}

func (client *HyperClient) ListPods() ([]HyperPod, error) {
	v := url.Values{}
	v.Set(KEY_ITEM, TYPE_POD)
	body, _, err := readBody(client.call("GET", "/list?"+v.Encode(), nil, nil))
	if err != nil {
		return nil, err
	}

	var podList map[string]interface{}
	err = json.Unmarshal(body, &podList)
	if err != nil {
		return nil, err
	}

	var result []HyperPod
	for _, pod := range podList["podData"].([]interface{}) {
		fields := strings.Split(pod.(string), ":")
		var hyperPod HyperPod
		hyperPod.podID = fields[0]
		hyperPod.podName = fields[1]
		hyperPod.vmName = fields[2]
		hyperPod.status = fields[3]

		values := url.Values{}
		values.Set(KEY_POD_NAME, hyperPod.podID)
		body, _, err = readBody(client.call("GET", "/pod/info?"+values.Encode(), nil, nil))
		if err != nil {
			return nil, err
		}

		err = json.Unmarshal(body, &hyperPod.podInfo)
		if err != nil {
			return nil, err
		}

		result = append(result, hyperPod)
	}

	return result, nil
}

func (client *HyperClient) ListContainers() ([]HyperContainer, error) {
	v := url.Values{}
	v.Set(KEY_ITEM, TYPE_CONTAINER)
	body, _, err := readBody(client.call("GET", "/list?"+v.Encode(), nil, nil))
	if err != nil {
		return nil, err
	}

	var containerList map[string]interface{}
	err = json.Unmarshal(body, &containerList)
	if err != nil {
		return nil, err
	}

	var result []HyperContainer
	for _, container := range containerList["cData"].([]interface{}) {
		fields := strings.Split(container.(string), ":")
		var h HyperContainer
		h.containerID = fields[0]
		if len(fields[1]) < 1 {
			return nil, errors.New("Hyper container name not resolved")
		}
		h.name = fields[1][1:]
		h.podID = fields[2]
		h.status = fields[3]

		result = append(result, h)
	}

	return result, nil
}

func (client *HyperClient) Info() (map[string]interface{}, error) {
	body, _, err := readBody(client.call("GET", "/info", nil, nil))
	if err != nil {
		return nil, err
	}

	var result map[string]interface{}
	err = json.Unmarshal(body, &result)
	if err != nil {
		return nil, err
	}

	return result, nil
}

func (client *HyperClient) ListImages() ([]HyperImage, error) {
	v := url.Values{}
	v.Set("all", "no")
	body, _, err := readBody(client.call("GET", "/images/get?"+v.Encode(), nil, nil))
	if err != nil {
		return nil, err
	}

	var images map[string][]string
	err = json.Unmarshal(body, &images)
	if err != nil {
		return nil, err
	}

	var hyperImages []HyperImage
	for _, image := range images["imagesList"] {
		imageDesc := strings.Split(image, ":")
		if len(imageDesc) != 5 {
			glog.Warning("Hyper: can not parse image info")
			return nil, fmt.Errorf("Hyper: can not parse image info")
		}

		var imageHyper HyperImage
		imageHyper.repository = imageDesc[0]
		imageHyper.tag = imageDesc[1]
		imageHyper.imageID = imageDesc[2]

		createdAt, err := strconv.ParseInt(imageDesc[3], 10, 0)
		if err != nil {
			return nil, err
		}
		imageHyper.createdAt = createdAt

		virtualSize, err := strconv.ParseInt(imageDesc[4], 10, 0)
		if err != nil {
			return nil, err
		}
		imageHyper.virtualSize = virtualSize

		hyperImages = append(hyperImages, imageHyper)
	}

	return hyperImages, nil
}

func (client *HyperClient) RemoveImage(imageID string) error {
	v := url.Values{}
	v.Set(KEY_IMAGEID, imageID)
	_, _, err := readBody(client.call("POST", "/images/remove?"+v.Encode(), nil, nil))
	if err != nil {
		return err
	}

	return nil
}

func (client *HyperClient) RemovePod(podID string) error {
	v := url.Values{}
	v.Set(KEY_POD_ID, podID)
	_, _, err := readBody(client.call("POST", "/pod/remove?"+v.Encode(), nil, nil))
	if err != nil {
		return err
	}

	return nil
}

func (client *HyperClient) StartPod(podID string) error {
	v := url.Values{}
	v.Set(KEY_POD_ID, podID)
	_, _, err := readBody(client.call("POST", "/pod/start?"+v.Encode(), nil, nil))
	if err != nil {
		return err
	}

	return nil
}

func (client *HyperClient) StopPod(podID string) error {
	v := url.Values{}
	v.Set(KEY_POD_ID, podID)
	v.Set("stopVM", "yes")
	_, _, err := readBody(client.call("POST", "/pod/stop?"+v.Encode(), nil, nil))
	if err != nil {
		return err
	}

	return nil
}

func (client *HyperClient) PullImage(image string, credential string) error {
	v := url.Values{}
	v.Set(KEY_IMAGENAME, image)

	headers := make(map[string][]string)
	if credential != "" {
		headers["X-Registry-Auth"] = []string{credential}
	}

	//_, _, err := readBody(client.call("POST", "/image/create?"+v.Encode(), nil, headers))
	err := client.stream("POST", "/image/create?"+v.Encode(), nil, nil, headers)
	if err != nil {
		return err
	}

	return nil
}

func (client *HyperClient) CreatePod(podArgs string) (map[string]interface{}, error) {
	glog.V(5).Infof("Hyper: starting to create pod %s", podArgs)
	v := url.Values{}
	v.Set(KEY_POD_ARGS, podArgs)
	body, _, err := readBody(client.call("POST", "/pod/create?"+v.Encode(), nil, nil))
	if err != nil {
		return nil, err
	}

	var result map[string]interface{}
	err = json.Unmarshal(body, &result)
	if err != nil {
		return nil, err
	}

	return result, nil
}

func (c *HyperClient) hijack(method, path string, hijackOptions hijackOptions) error {
	var params io.Reader
	if hijackOptions.data != nil {
		buf, err := json.Marshal(hijackOptions.data)
		if err != nil {
			return err
		}
		params = bytes.NewBuffer(buf)
	}

	if hijackOptions.stdout == nil {
		hijackOptions.stdout = ioutil.Discard
	}
	if hijackOptions.stderr == nil {
		hijackOptions.stderr = ioutil.Discard
	}
	req, err := http.NewRequest(method, fmt.Sprintf("/v%s%s", HYPER_MINVERSION, path), params)
	if err != nil {
		return err
	}

	req.Header.Set("User-Agent", "kubelet")
	req.Header.Set("Content-Type", "text/plain")
	req.Header.Set("Connection", "Upgrade")
	req.Header.Set("Upgrade", "tcp")
	req.Host = HYPER_ADDR

	dial, err := net.Dial(HYPER_PROTO, HYPER_ADDR)
	if err != nil {
		return err
	}

	clientconn := httputil.NewClientConn(dial, nil)
	defer clientconn.Close()
	clientconn.Do(req)
	if hijackOptions.success != nil {
		hijackOptions.success <- struct{}{}
		<-hijackOptions.success
	}
	rwc, br := clientconn.Hijack()
	defer rwc.Close()
	errChanOut := make(chan error, 1)
	errChanIn := make(chan error, 1)
	exit := make(chan bool)
	go func() {
		defer close(exit)
		defer close(errChanOut)
		var err error
		if hijackOptions.setRawTerminal {
			// When TTY is ON, use regular copy
			_, err = io.Copy(hijackOptions.stdout, br)
		} else {
			_, err = stdcopy.StdCopy(hijackOptions.stdout, hijackOptions.stderr, br)
		}
		errChanOut <- err
	}()
	go func() {
		if hijackOptions.in != nil {
			_, err := io.Copy(rwc, hijackOptions.in)
			errChanIn <- err
		}
		rwc.(interface {
			CloseWrite() error
		}).CloseWrite()
	}()
	<-exit
	select {
	case err = <-errChanIn:
		return err
	case err = <-errChanOut:
		return err
	}
}

func (client *HyperClient) Attach(opts AttachToContainerOptions) error {
	if opts.Container == "" {
		return fmt.Errorf("No Such Container %s", opts.Container)
	}

	v := url.Values{}
	v.Set(KEY_TYPE, TYPE_CONTAINER)
	v.Set(KEY_VALUE, opts.Container)
	path := "/attach?" + v.Encode()
	return client.hijack("POST", path, hijackOptions{
		success:        opts.Success,
		setRawTerminal: opts.RawTerminal,
		in:             opts.InputStream,
		stdout:         opts.OutputStream,
		stderr:         opts.ErrorStream,
	})
}

func (client *HyperClient) IsImagePresent(repo, tag string) (bool, error) {
	if outputs, err := client.ListImages(); err == nil {
		for _, imgInfo := range outputs {
			if imgInfo.repository == repo && imgInfo.tag == tag {
				return true, nil
			}
		}
	}
	return false, nil
}
