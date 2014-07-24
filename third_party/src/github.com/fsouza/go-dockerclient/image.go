// Copyright 2014 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"time"
)

// APIImages represent an image returned in the ListImages call.
type APIImages struct {
	ID          string   `json:"Id"`
	RepoTags    []string `json:",omitempty"`
	Created     int64
	Size        int64
	VirtualSize int64
	ParentId    string `json:",omitempty"`
	Repository  string `json:",omitempty"`
	Tag         string `json:",omitempty"`
}

type Image struct {
	ID              string    `json:"id"`
	Parent          string    `json:"parent,omitempty"`
	Comment         string    `json:"comment,omitempty"`
	Created         time.Time `json:"created"`
	Container       string    `json:"container,omitempty"`
	ContainerConfig Config    `json:"containerconfig,omitempty"`
	DockerVersion   string    `json:"dockerversion,omitempty"`
	Author          string    `json:"author,omitempty"`
	Config          *Config   `json:"config,omitempty"`
	Architecture    string    `json:"architecture,omitempty"`
	Size            int64
}

type ImagePre012 struct {
	ID              string    `json:"id"`
	Parent          string    `json:"parent,omitempty"`
	Comment         string    `json:"comment,omitempty"`
	Created         time.Time `json:"created"`
	Container       string    `json:"container,omitempty"`
	ContainerConfig Config    `json:"container_config,omitempty"`
	DockerVersion   string    `json:"docker_version,omitempty"`
	Author          string    `json:"author,omitempty"`
	Config          *Config   `json:"config,omitempty"`
	Architecture    string    `json:"architecture,omitempty"`
	Size            int64
}

var (
	// ErrNoSuchImage is the error returned when the image does not exist.
	ErrNoSuchImage = errors.New("no such image")

	// ErrMissingRepo is the error returned when the remote repository is
	// missing.
	ErrMissingRepo = errors.New("missing remote repository e.g. 'github.com/user/repo'")

	// ErrMissingOutputStream is the error returned when no output stream
	// is provided to some calls, like BuildImage.
	ErrMissingOutputStream = errors.New("missing output stream")
)

// ListImages returns the list of available images in the server.
//
// See http://goo.gl/dkMrwP for more details.
func (c *Client) ListImages(all bool) ([]APIImages, error) {
	path := "/images/json?all="
	if all {
		path += "1"
	} else {
		path += "0"
	}
	body, _, err := c.do("GET", path, nil)
	if err != nil {
		return nil, err
	}
	var images []APIImages
	err = json.Unmarshal(body, &images)
	if err != nil {
		return nil, err
	}
	return images, nil
}

// RemoveImage removes an image by its name or ID.
//
// See http://goo.gl/7hjHHy for more details.
func (c *Client) RemoveImage(name string) error {
	_, status, err := c.do("DELETE", "/images/"+name, nil)
	if status == http.StatusNotFound {
		return ErrNoSuchImage
	}
	return err
}

// InspectImage returns an image by its name or ID.
//
// See http://goo.gl/pHEbma for more details.
func (c *Client) InspectImage(name string) (*Image, error) {
	body, status, err := c.do("GET", "/images/"+name+"/json", nil)
	if status == http.StatusNotFound {
		return nil, ErrNoSuchImage
	}
	if err != nil {
		return nil, err
	}

	var image Image

	// if the caller elected to skip checking the server's version, assume it's the latest
	if c.SkipServerVersionCheck || c.expectedApiVersion.GreaterThanOrEqualTo(apiVersion_1_12) {
		err = json.Unmarshal(body, &image)
		if err != nil {
			return nil, err
		}
	} else {
		var imagePre012 ImagePre012
		err = json.Unmarshal(body, &imagePre012)
		if err != nil {
			return nil, err
		}

		image.ID = imagePre012.ID
		image.Parent = imagePre012.Parent
		image.Comment = imagePre012.Comment
		image.Created = imagePre012.Created
		image.Container = imagePre012.Container
		image.ContainerConfig = imagePre012.ContainerConfig
		image.DockerVersion = imagePre012.DockerVersion
		image.Author = imagePre012.Author
		image.Config = imagePre012.Config
		image.Architecture = imagePre012.Architecture
		image.Size = imagePre012.Size
	}

	return &image, nil
}

// PushImageOptions represents options to use in the PushImage method.
//
// See http://goo.gl/GBmyhc for more details.
type PushImageOptions struct {
	// Name of the image
	Name string

	// Tag of the image
	Tag string

	// Registry server to push the image
	Registry string

	OutputStream io.Writer `qs:"-"`
}

// AuthConfiguration represents authentication options to use in the PushImage
// method. It represents the authencation in the Docker index server.
type AuthConfiguration struct {
	Username string `json:"username,omitempty"`
	Password string `json:"password,omitempty"`
	Email    string `json:"email,omitempty"`
}

// PushImage pushes an image to a remote registry, logging progress to w.
//
// An empty instance of AuthConfiguration may be used for unauthenticated
// pushes.
//
// See http://goo.gl/GBmyhc for more details.
func (c *Client) PushImage(opts PushImageOptions, auth AuthConfiguration) error {
	if opts.Name == "" {
		return ErrNoSuchImage
	}
	name := opts.Name
	opts.Name = ""
	path := "/images/" + name + "/push?" + queryString(&opts)
	var headers = make(map[string]string)
	var buf bytes.Buffer
	json.NewEncoder(&buf).Encode(auth)

	headers["X-Registry-Auth"] = base64.URLEncoding.EncodeToString(buf.Bytes())

	return c.stream("POST", path, true, headers, nil, opts.OutputStream, nil)
}

// PullImageOptions present the set of options available for pulling an image
// from a registry.
//
// See http://goo.gl/PhBKnS for more details.
type PullImageOptions struct {
	Repository   string `qs:"fromImage"`
	Registry     string
	Tag          string
	OutputStream io.Writer `qs:"-"`
}

// PullImage pulls an image from a remote registry, logging progress to w.
//
// See http://goo.gl/PhBKnS for more details.
func (c *Client) PullImage(opts PullImageOptions, auth AuthConfiguration) error {
	if opts.Repository == "" {
		return ErrNoSuchImage
	}

	var headers = make(map[string]string)
	var buf bytes.Buffer
	json.NewEncoder(&buf).Encode(auth)
	headers["X-Registry-Auth"] = base64.URLEncoding.EncodeToString(buf.Bytes())

	return c.createImage(queryString(&opts), headers, nil, opts.OutputStream)
}

func (c *Client) createImage(qs string, headers map[string]string, in io.Reader, w io.Writer) error {
	path := "/images/create?" + qs
	return c.stream("POST", path, true, headers, in, w, nil)
}

// ImportImageOptions present the set of informations available for importing
// an image from a source file or the stdin.
//
// See http://goo.gl/PhBKnS for more details.
type ImportImageOptions struct {
	Repository string `qs:"repo"`
	Source     string `qs:"fromSrc"`
	Tag        string `qs:"tag"`

	InputStream  io.Reader `qs:"-"`
	OutputStream io.Writer `qs:"-"`
}

// ImportImage imports an image from a url, a file or stdin
//
// See http://goo.gl/PhBKnS for more details.
func (c *Client) ImportImage(opts ImportImageOptions) error {
	if opts.Repository == "" {
		return ErrNoSuchImage
	}
	if opts.Source != "-" {
		opts.InputStream = nil
	}
	if opts.Source != "-" && !isURL(opts.Source) {
		f, err := os.Open(opts.Source)
		if err != nil {
			return err
		}
		b, err := ioutil.ReadAll(f)
		opts.InputStream = bytes.NewBuffer(b)
		opts.Source = "-"
	}
	return c.createImage(queryString(&opts), nil, opts.InputStream, opts.OutputStream)
}

// BuildImageOptions present the set of informations available for building
// an image from a tarfile with a Dockerfile in it,the details about Dockerfile
// see http://docs.docker.io/en/latest/reference/builder/
type BuildImageOptions struct {
	Name           string    `qs:"t"`
	NoCache        bool      `qs:"nocache"`
	SuppressOutput bool      `qs:"q"`
	RmTmpContainer bool      `qs:"rm"`
	InputStream    io.Reader `qs:"-"`
	OutputStream   io.Writer `qs:"-"`
	Remote         string    `qs:"remote"`
}

// BuildImage builds an image from a tarball's url or a Dockerfile in the input
// stream.
func (c *Client) BuildImage(opts BuildImageOptions) error {
	if opts.OutputStream == nil {
		return ErrMissingOutputStream
	}
	var headers map[string]string
	if opts.Remote != "" && opts.Name == "" {
		opts.Name = opts.Remote
	}
	if opts.InputStream != nil {
		headers = map[string]string{"Content-Type": "application/tar"}
	} else if opts.Remote == "" {
		return ErrMissingRepo
	}
	return c.stream("POST", fmt.Sprintf("/build?%s",
		queryString(&opts)), true, headers, opts.InputStream, opts.OutputStream, nil)
}

// TagImageOptions present the set of options to tag an image
type TagImageOptions struct {
	Repo  string
	Tag   string
	Force bool
}

// TagImage adds a tag to the image 'name'
func (c *Client) TagImage(name string, opts TagImageOptions) error {
	if name == "" {
		return ErrNoSuchImage
	}
	_, status, err := c.do("POST", fmt.Sprintf("/images/"+name+"/tag?%s",
		queryString(&opts)), nil)
	if status == http.StatusNotFound {
		return ErrNoSuchImage
	}

	return err
}

func isURL(u string) bool {
	p, err := url.Parse(u)
	if err != nil {
		return false
	}
	return p.Scheme == "http" || p.Scheme == "https"
}
