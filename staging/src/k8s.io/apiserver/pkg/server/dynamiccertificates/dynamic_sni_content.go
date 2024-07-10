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

package dynamiccertificates

import (
	"k8s.io/klog/v2"
)

// DynamicFileSNIContent provides a SNICertKeyContentProvider that can dynamically react to new file content
type DynamicFileSNIContent struct {
	*DynamicCertKeyPairContent
	sniNames []string
}

var _ SNICertKeyContentProvider = &DynamicFileSNIContent{}
var _ ControllerRunner = &DynamicFileSNIContent{}

// NewDynamicSNIContentFromFiles returns a dynamic SNICertKeyContentProvider based on a cert and key filename and explicit names.
// It uses a default logger (klog.Background) for logging.
func NewDynamicSNIContentFromFiles(purpose, certFile, keyFile string, sniNames ...string) (*DynamicFileSNIContent, error) {
	logger := klog.Background()

	return NewDynamicSNIContentFromFilesWithLogger(logger, purpose, certFile, keyFile, sniNames...)
}

// NewDynamicSNIContentFromFilesWithLogger returns a dynamic SNICertKeyContentProvider based on a cert and key filename and explicit names
// It uses the provided logger for logging.
func NewDynamicSNIContentFromFilesWithLogger(logger klog.Logger, purpose, certFile, keyFile string, sniNames ...string) (*DynamicFileSNIContent, error) {
	servingContent, err := NewDynamicServingContentFromFilesWithLogger(logger, purpose, certFile, keyFile)

	if err != nil {
		return nil, err
	}

	ret := &DynamicFileSNIContent{
		DynamicCertKeyPairContent: servingContent,
		sniNames:                  sniNames,
	}
	if err := ret.loadCertKeyPair(logger); err != nil {
		return nil, err
	}

	return ret, nil
}

// SNINames returns explicitly set SNI names for the certificate. These are not dynamic.
func (c *DynamicFileSNIContent) SNINames() []string {
	return c.sniNames
}
