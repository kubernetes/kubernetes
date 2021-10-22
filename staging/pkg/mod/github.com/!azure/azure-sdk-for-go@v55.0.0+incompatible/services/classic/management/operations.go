// +build go1.7

package management

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"encoding/xml"
	"errors"
	"fmt"
	"time"
)

var (
	// ErrOperationCancelled from WaitForOperation when the polling loop is
	// cancelled through signaling the channel.
	ErrOperationCancelled = errors.New("Polling for operation status cancelled")
)

// GetOperationStatusResponse represents an in-flight operation. Use
// client.GetOperationStatus() to get the operation given the operation ID, or
// use WaitForOperation() to poll and wait until the operation has completed.
// See https://msdn.microsoft.com/en-us/library/azure/ee460783.aspx
type GetOperationStatusResponse struct {
	XMLName        xml.Name `xml:"http://schemas.microsoft.com/windowsazure Operation"`
	ID             string
	Status         OperationStatus
	HTTPStatusCode string
	Error          *AzureError
}

// OperationStatus describes the states an Microsoft Azure Service Management
// operation an be in.
type OperationStatus string

// List of states an operation can be reported as
const (
	OperationStatusInProgress OperationStatus = "InProgress"
	OperationStatusSucceeded  OperationStatus = "Succeeded"
	OperationStatusFailed     OperationStatus = "Failed"
)

// OperationID is assigned by Azure API and can be used to look up the status of
// an operation
type OperationID string

func (c client) GetOperationStatus(operationID OperationID) (GetOperationStatusResponse, error) {
	operation := GetOperationStatusResponse{}
	if operationID == "" {
		return operation, fmt.Errorf(errParamNotSpecified, "operationID")
	}

	url := fmt.Sprintf("operations/%s", operationID)
	response, azureErr := c.SendAzureGetRequest(url)
	if azureErr != nil {
		return operation, azureErr
	}

	err := xml.Unmarshal(response, &operation)
	return operation, err
}

func (c client) WaitForOperation(operationID OperationID, cancel chan struct{}) error {
	for {
		done, err := c.checkOperationStatus(operationID)
		if err != nil || done {
			return err
		}
		select {
		case <-time.After(c.config.OperationPollInterval):
		case <-cancel:
			return ErrOperationCancelled
		}
	}
}

func (c client) checkOperationStatus(id OperationID) (done bool, err error) {
	op, err := c.GetOperationStatus(id)
	if err != nil {
		return false, fmt.Errorf("Failed to get operation status '%s': %v", id, err)
	}

	switch op.Status {
	case OperationStatusSucceeded:
		return true, nil
	case OperationStatusFailed:
		if op.Error != nil {
			return true, op.Error
		}
		return true, fmt.Errorf("Azure Operation (x-ms-request-id=%s) has failed", id)
	case OperationStatusInProgress:
		return false, nil
	default:
		return false, fmt.Errorf("Unknown operation status returned from API: %s (x-ms-request-id=%s)", op.Status, id)
	}
}
