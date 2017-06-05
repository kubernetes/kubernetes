/*
Copyright 2014 The Kubernetes Authors.

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
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ec2"
	"strings"
	"testing"
)

func TestFilterTags(t *testing.T) {
	awsServices := NewFakeAWSServices()
	c, err := newAWSCloud(strings.NewReader("[global]"), awsServices)
	if err != nil {
		t.Errorf("Error building aws cloud: %v", err)
		return
	}

	if c.tagging.ClusterID != TestClusterId {
		t.Errorf("unexpected ClusterID: %v", c.tagging.ClusterID)
	}
}

func TestFindClusterID(t *testing.T) {
	grid := []struct {
		Tags           map[string]string
		ExpectedNew    string
		ExpectedLegacy string
		ExpectError    bool
	}{
		{
			Tags: map[string]string{},
		},
		{
			Tags: map[string]string{
				TagNameKubernetesClusterLegacy: "a",
			},
			ExpectedLegacy: "a",
		},
		{
			Tags: map[string]string{
				TagNameKubernetesClusterPrefix + "a": "owned",
			},
			ExpectedNew: "a",
		},
		{
			Tags: map[string]string{
				TagNameKubernetesClusterPrefix + "a": "",
			},
			ExpectedNew: "a",
		},
		{
			Tags: map[string]string{
				TagNameKubernetesClusterLegacy:       "a",
				TagNameKubernetesClusterPrefix + "a": "",
			},
			ExpectedLegacy: "a",
			ExpectedNew:    "a",
		},
		{
			Tags: map[string]string{
				TagNameKubernetesClusterPrefix + "a": "",
				TagNameKubernetesClusterPrefix + "b": "",
			},
			ExpectError: true,
		},
	}
	for _, g := range grid {
		var ec2Tags []*ec2.Tag
		for k, v := range g.Tags {
			ec2Tags = append(ec2Tags, &ec2.Tag{Key: aws.String(k), Value: aws.String(v)})
		}
		actualLegacy, actualNew, err := findClusterIDs(ec2Tags)
		if g.ExpectError {
			if err == nil {
				t.Errorf("expected error for tags %v", g.Tags)
				continue
			}
		} else {
			if err != nil {
				t.Errorf("unexpected error for tags %v: %v", g.Tags, err)
				continue
			}

			if g.ExpectedNew != actualNew {
				t.Errorf("unexpected new clusterid for tags %v: %s vs %s", g.Tags, g.ExpectedNew, actualNew)
				continue
			}

			if g.ExpectedLegacy != actualLegacy {
				t.Errorf("unexpected new clusterid for tags %v: %s vs %s", g.Tags, g.ExpectedLegacy, actualLegacy)
				continue
			}
		}
	}
}
