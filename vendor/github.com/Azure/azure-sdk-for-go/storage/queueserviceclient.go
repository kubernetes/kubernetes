package storage

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

// QueueServiceClient contains operations for Microsoft Azure Queue Storage
// Service.
type QueueServiceClient struct {
	client Client
	auth   authentication
}

// GetServiceProperties gets the properties of your storage account's queue service.
// See: https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/get-queue-service-properties
func (q *QueueServiceClient) GetServiceProperties() (*ServiceProperties, error) {
	return q.client.getServiceProperties(queueServiceName, q.auth)
}

// SetServiceProperties sets the properties of your storage account's queue service.
// See: https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/set-queue-service-properties
func (q *QueueServiceClient) SetServiceProperties(props ServiceProperties) error {
	return q.client.setServiceProperties(props, queueServiceName, q.auth)
}

// GetQueueReference returns a Container object for the specified queue name.
func (q *QueueServiceClient) GetQueueReference(name string) *Queue {
	return &Queue{
		qsc:  q,
		Name: name,
	}
}
