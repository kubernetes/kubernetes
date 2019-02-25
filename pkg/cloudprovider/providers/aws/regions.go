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

package aws

import (
	"sync"

	"k8s.io/klog"

	"k8s.io/apimachinery/pkg/util/sets"
	awscredentialprovider "k8s.io/kubernetes/pkg/credentialprovider/aws"
)

// wellKnownRegions is the complete list of regions known to the AWS cloudprovider
// and credentialprovider.
var wellKnownRegions = [...]string{
	// from `aws ec2 describe-regions --region us-east-1 --query Regions[].RegionName | sort`
	"ap-northeast-1",
	"ap-northeast-2",
	"ap-northeast-3",
	"ap-south-1",
	"ap-southeast-1",
	"ap-southeast-2",
	"ca-central-1",
	"eu-central-1",
	"eu-west-1",
	"eu-west-2",
	"eu-west-3",
	"sa-east-1",
	"us-east-1",
	"us-east-2",
	"us-west-1",
	"us-west-2",

	// these are not registered in many / most accounts
	"cn-north-1",
	"cn-northwest-1",
	"us-gov-west-1",
}

// awsRegionsMutex protects awsRegions
var awsRegionsMutex sync.Mutex

// awsRegions is a set of recognized regions
var awsRegions sets.String

// recognizeRegion is called for each AWS region we know about.
// It currently registers a credential provider for that region.
// There are two paths to discovering a region:
//  * we hard-code some well-known regions
//  * if a region is discovered from instance metadata, we add that
func recognizeRegion(region string) {
	awsRegionsMutex.Lock()
	defer awsRegionsMutex.Unlock()

	if awsRegions == nil {
		awsRegions = sets.NewString()
	}

	if awsRegions.Has(region) {
		klog.V(6).Infof("found AWS region %q again - ignoring", region)
		return
	}

	klog.V(4).Infof("found AWS region %q", region)

	awscredentialprovider.RegisterCredentialsProvider(region)

	awsRegions.Insert(region)
}

// recognizeWellKnownRegions calls RecognizeRegion on each WellKnownRegion
func recognizeWellKnownRegions() {
	for _, region := range wellKnownRegions {
		recognizeRegion(region)
	}
}

// isRegionValid checks if the region is in the set of known regions
func isRegionValid(region string) bool {
	awsRegionsMutex.Lock()
	defer awsRegionsMutex.Unlock()

	return awsRegions.Has(region)
}
