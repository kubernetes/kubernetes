// +build !providerless

/*
Copyright 2015 The Kubernetes Authors.

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

package aws

import (
	"github.com/aws/aws-sdk-go/aws/request"
	"k8s.io/klog/v2"
)

// Handler for aws-sdk-go that logs all requests
func awsHandlerLogger(req *request.Request) {
	service, name := awsServiceAndName(req)
	klog.V(4).InfoS("AWS request", "service", service, "name", name)
}

func awsSendHandlerLogger(req *request.Request) {
	service, name := awsServiceAndName(req)
	klog.V(4).Infof("AWS API Send", "service", service, "name", name, "Operation", req.Operation, "Params", req.Params)
}

func awsValidateResponseHandlerLogger(req *request.Request) {
	service, name := awsServiceAndName(req)
	klog.V(4).Infof("AWS API ValidateResponse", "service", service, "name", name, "Operation", req.Operation, "Params", req.Params, "Status",req.HTTPResponse.Status)
}

func awsServiceAndName(req *request.Request) (string, string) {
	service := req.ClientInfo.ServiceName

	name := "?"
	if req.Operation != nil {
		name = req.Operation.Name
	}
	return service, name
}
