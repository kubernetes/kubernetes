package storage

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

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
