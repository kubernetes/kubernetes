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
	"encoding/json"
	"time"
)

// Contains functionality for tasks API.
type TasksAPI struct {
	client *Client
}

var taskUrl string = rootUrl + "/tasks"

// Gets a task by ID.
func (api *TasksAPI) Get(id string) (task *Task, err error) {
	res, err := api.client.restClient.Get(api.client.Endpoint+taskUrl+"/"+id, api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	result, err := getTask(getError(res))
	return result, err
}

// Gets all tasks, using options to filter the results.
// If options is nil, no filtering will occur.
func (api *TasksAPI) GetAll(options *TaskGetOptions) (result *TaskList, err error) {
	uri := api.client.Endpoint + taskUrl
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

// Waits for a task to complete by polling the tasks API until a task returns with
// the state COMPLETED or ERROR. Will wait no longer than the duration specified by timeout.
func (api *TasksAPI) WaitTimeout(id string, timeout time.Duration) (task *Task, err error) {
	start := time.Now()
	numErrors := 0
	maxErrors := api.client.options.TaskRetryCount

	for time.Since(start) < timeout {
		task, err = api.Get(id)
		if err != nil {
			switch err.(type) {
			// If an ApiError comes back, something is wrong, return the error to the caller
			case ApiError:
				return
			// For other errors, retry before giving up
			default:
				numErrors++
				if numErrors > maxErrors {
					return
				}
			}
		} else {
			// Reset the error count any time a successful call is made
			numErrors = 0
			if task.State == "COMPLETED" {
				return
			}
			if task.State == "ERROR" {
				err = TaskError{task.ID, getFailedStep(task)}
				return
			}
		}
		time.Sleep(api.client.options.TaskPollDelay)
	}
	err = TaskTimeoutError{id}
	return
}

// Waits for a task to complete by polling the tasks API until a task returns with
// the state COMPLETED or ERROR.
func (api *TasksAPI) Wait(id string) (task *Task, err error) {
	return api.WaitTimeout(id, api.client.options.TaskPollTimeout)
}

// Gets the failed step in the task to get error details for failed task.
func getFailedStep(task *Task) (step Step) {
	var errorStep Step
	for _, s := range task.Steps {
		if s.State == "ERROR" {
			errorStep = s
			break
		}
	}

	return errorStep
}
