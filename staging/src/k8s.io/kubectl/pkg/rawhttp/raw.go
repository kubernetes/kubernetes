/*
Copyright 2019 The Kubernetes Authors.

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

package rawhttp

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"io/ioutil"
	"os"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/rest"
)

// RawPost uses the REST client to POST content
func RawPost(restClient *rest.RESTClient, streams genericclioptions.IOStreams, url, filename string) error {
	return raw(restClient, streams, url, filename, "POST")
}

// RawPut uses the REST client to PUT content
func RawPut(restClient *rest.RESTClient, streams genericclioptions.IOStreams, url, filename string) error {
	return raw(restClient, streams, url, filename, "PUT")
}

// RawGet uses the REST client to GET content
func RawGet(restClient *rest.RESTClient, streams genericclioptions.IOStreams, url string) error {
	return raw(restClient, streams, url, "", "GET")
}

// RawDelete uses the REST client to DELETE content
func RawDelete(restClient *rest.RESTClient, streams genericclioptions.IOStreams, url, filename string) error {
	return raw(restClient, streams, url, filename, "DELETE")
}

// raw makes a simple HTTP request to the provided path on the server using the default credentials.
func raw(restClient *rest.RESTClient, streams genericclioptions.IOStreams, url, filename, requestType string) error {
	var data io.ReadCloser
	switch {
	case len(filename) == 0:
		data = ioutil.NopCloser(bytes.NewBuffer([]byte{}))

	case filename == "-":
		data = ioutil.NopCloser(streams.In)

	default:
		var err error
		data, err = os.Open(filename)
		if err != nil {
			return err
		}
	}

	var request *rest.Request
	switch requestType {
	case "GET":
		request = restClient.Get().RequestURI(url)
	case "PUT":
		request = restClient.Put().RequestURI(url).Body(data)
	case "POST":
		request = restClient.Post().RequestURI(url).Body(data)
	case "DELETE":
		request = restClient.Delete().RequestURI(url).Body(data)

	default:
		return fmt.Errorf("unknown requestType: %q", requestType)
	}

	stream, err := request.Stream(context.TODO())
	if err != nil {
		return err
	}
	defer stream.Close()

	_, err = io.Copy(streams.Out, stream)
	if err != nil && err != io.EOF {
		return err
	}
	return nil
}
