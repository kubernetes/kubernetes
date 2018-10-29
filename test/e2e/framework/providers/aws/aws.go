/*
Copyright 2018 The Kubernetes Authors.

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
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/autoscaling"
	"github.com/aws/aws-sdk-go/service/ec2"

	"k8s.io/api/core/v1"
	awscloud "k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	"k8s.io/kubernetes/test/e2e/framework"
)

func init() {
	framework.RegisterProvider("aws", NewProvider)
}

func NewProvider() (framework.ProviderInterface, error) {
	if framework.TestContext.CloudConfig.Zone == "" {
		return nil, fmt.Errorf("gce-zone must be specified for AWS")
	}
	return &Provider{}, nil
}

type Provider struct {
	framework.NullProvider
}

func (p *Provider) ResizeGroup(group string, size int32) error {
	client := autoscaling.New(session.New())
	return awscloud.ResizeInstanceGroup(client, group, int(size))
}

func (p *Provider) GroupSize(group string) (int, error) {
	client := autoscaling.New(session.New())
	instanceGroup, err := awscloud.DescribeInstanceGroup(client, group)
	if err != nil {
		return -1, fmt.Errorf("error describing instance group: %v", err)
	}
	if instanceGroup == nil {
		return -1, fmt.Errorf("instance group not found: %s", group)
	}
	return instanceGroup.CurrentSize()
}

func (p *Provider) CreatePD(zone string) (string, error) {
	client := newAWSClient(zone)
	request := &ec2.CreateVolumeInput{}
	request.AvailabilityZone = aws.String(zone)
	request.Size = aws.Int64(10)
	request.VolumeType = aws.String(awscloud.DefaultVolumeType)
	response, err := client.CreateVolume(request)
	if err != nil {
		return "", err
	}

	az := aws.StringValue(response.AvailabilityZone)
	awsID := aws.StringValue(response.VolumeId)

	volumeName := "aws://" + az + "/" + awsID
	return volumeName, nil
}

func (p *Provider) DeletePD(pdName string) error {
	client := newAWSClient("")

	tokens := strings.Split(pdName, "/")
	awsVolumeID := tokens[len(tokens)-1]

	request := &ec2.DeleteVolumeInput{VolumeId: aws.String(awsVolumeID)}
	_, err := client.DeleteVolume(request)
	if err != nil {
		if awsError, ok := err.(awserr.Error); ok && awsError.Code() == "InvalidVolume.NotFound" {
			framework.Logf("volume deletion implicitly succeeded because volume %q does not exist.", pdName)
		} else {
			return fmt.Errorf("error deleting EBS volumes: %v", err)
		}
	}
	return nil
}

func (p *Provider) CreatePVSource(zone, diskName string) (*v1.PersistentVolumeSource, error) {
	return &v1.PersistentVolumeSource{
		AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
			VolumeID: diskName,
			FSType:   "ext3",
		},
	}, nil
}

func (p *Provider) DeletePVSource(pvSource *v1.PersistentVolumeSource) error {
	return framework.DeletePDWithRetry(pvSource.AWSElasticBlockStore.VolumeID)
}

func newAWSClient(zone string) *ec2.EC2 {
	var cfg *aws.Config

	if zone == "" {
		zone = framework.TestContext.CloudConfig.Zone
	}
	if zone == "" {
		framework.Logf("Warning: No AWS zone configured!")
		cfg = nil
	} else {
		region := zone[:len(zone)-1]
		cfg = &aws.Config{Region: aws.String(region)}
	}
	return ec2.New(session.New(), cfg)
}
