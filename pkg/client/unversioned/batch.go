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

package unversioned

import (
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/client/restclient"
)

type BatchInterface interface {
	JobsNamespacer
	ScheduledJobsNamespacer
}

// BatchClient is used to interact with Kubernetes batch features.
type BatchClient struct {
	*restclient.RESTClient
}

func (c *BatchClient) Jobs(namespace string) JobInterface {
	return newJobsV1(c, namespace)
}

func (c *BatchClient) ScheduledJobs(namespace string) ScheduledJobInterface {
	return newScheduledJobs(c, namespace)
}

func NewBatch(c *restclient.Config) (*BatchClient, error) {
	config := *c
	if err := setGroupDefaults(batch.GroupName, &config); err != nil {
		return nil, err
	}
	client, err := restclient.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &BatchClient{client}, nil
}

func NewBatchOrDie(c *restclient.Config) *BatchClient {
	client, err := NewBatch(c)
	if err != nil {
		panic(err)
	}
	return client
}
