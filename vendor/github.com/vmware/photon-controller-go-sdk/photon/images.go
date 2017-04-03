// Copyright (c) 2016 VMware, Inc. All Rights Reserved.
//
// This product is licensed to you under the Apache License, Version 2.0 (the "License").
// You may not use this product except in compliance with the License.
//
// This product may include a number of subcomponents with separate copyright notices and
// license terms. Your use of these subcomponents is subject to the terms and conditions
// of the subcomponent's license, as noted in the LICENSE file.

package photon

import (
	"bytes"
	"encoding/json"
	"io"
)

// Contains functionality for images API.
type ImagesAPI struct {
	client *Client
}

// Options for GetImage API.
type ImageGetOptions struct {
	Name string `urlParam:"name"`
}

var imageUrl string = rootUrl + "/images"

// Uploads a new image, reading from the specified image path.
// If options is nil, default options are used.
func (api *ImagesAPI) CreateFromFile(imagePath string, options *ImageCreateOptions) (task *Task, err error) {
	params := imageCreateOptionsToMap(options)
	res, err := api.client.restClient.MultipartUploadFile(api.client.Endpoint+imageUrl, imagePath, params, api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	result, err := getTask(getError(res))
	return result, err
}

// Uploads a new image, reading from the specified io.Reader.
// Name is a descriptive name of the image, it is used in the filename field of the Content-Disposition header,
// and does not need to be unique.
// If options is nil, default options are used.
func (api *ImagesAPI) Create(reader io.ReadSeeker, name string, options *ImageCreateOptions) (task *Task, err error) {
	params := imageCreateOptionsToMap(options)
	res, err := api.client.restClient.MultipartUpload(api.client.Endpoint+imageUrl, reader, name, params, api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	result, err := getTask(getError(res))
	return result, err
}

// Gets all images on this photon instance.
func (api *ImagesAPI) GetAll(options *ImageGetOptions) (images *Images, err error) {
	uri := api.client.Endpoint + imageUrl
	if options != nil {
		uri += getQueryString(options)
	}
	res, err := api.client.restClient.GetList(api.client.Endpoint, uri, api.client.options.TokenOptions)
	if err != nil {
		return
	}

	images = &Images{}
	err = json.Unmarshal(res, images)
	return
}

// Gets details of image with the specified ID.
func (api *ImagesAPI) Get(imageID string) (image *Image, err error) {
	res, err := api.client.restClient.Get(api.client.Endpoint+imageUrl+"/"+imageID, api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	res, err = getError(res)
	if err != nil {
		return
	}
	var result Image
	err = json.NewDecoder(res.Body).Decode(&result)
	return &result, nil
}

// Deletes image with the specified ID.
func (api *ImagesAPI) Delete(imageID string) (task *Task, err error) {
	res, err := api.client.restClient.Delete(api.client.Endpoint+imageUrl+"/"+imageID, api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	result, err := getTask(getError(res))
	return result, err
}

// Gets all tasks with the specified image ID, using options to filter the results.
// If options is nil, no filtering will occur.
func (api *ImagesAPI) GetTasks(id string, options *TaskGetOptions) (result *TaskList, err error) {
	uri := api.client.Endpoint + imageUrl + "/" + id + "/tasks"
	if options != nil {
		uri += getQueryString(options)
	}

	res, err := api.client.restClient.GetList(api.client.Endpoint, uri, api.client.options.TokenOptions)
	if err != nil {
		return
	}

	result = &TaskList{}
	err = json.Unmarshal(res, result)
	return
}

// Gets IAM Policy of an image.
func (api *ImagesAPI) GetIam(imageID string) (policy *[]PolicyEntry, err error) {
	res, err := api.client.restClient.Get(
		api.client.Endpoint+imageUrl+"/"+imageID+"/iam",
		api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	res, err = getError(res)
	if err != nil {
		return
	}
	var result []PolicyEntry
	err = json.NewDecoder(res.Body).Decode(&result)
	return &result, nil
}

// Sets IAM Policy on an image.
func (api *ImagesAPI) SetIam(imageID string, policy *[]PolicyEntry) (task *Task, err error) {
	body, err := json.Marshal(policy)
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+imageUrl+"/"+imageID+"/iam",
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

// Modifies IAM Policy on an image.
func (api *ImagesAPI) ModifyIam(imageID string, policyDelta *PolicyDelta) (task *Task, err error) {
	body, err := json.Marshal(policyDelta)
	if err != nil {
		return
	}
	res, err := api.client.restClient.Put(
		api.client.Endpoint+imageUrl+"/"+imageID+"/iam",
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

func imageCreateOptionsToMap(opts *ImageCreateOptions) map[string]string {
	if opts == nil {
		return nil
	}
	return map[string]string{
		"ImageReplication": opts.ReplicationType,
	}
}
