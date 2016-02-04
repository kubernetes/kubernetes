/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package aws_credentials

import (
	"encoding/base64"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ecr"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/credentialprovider"
)

var registryUrls = []string{"*.dkr.ecr.*.amazonaws.com"}

// awsHandlerLogger is a handler that logs all AWS SDK requests
// Copied from cloudprovider/aws/log_handler.go
func awsHandlerLogger(req *request.Request) {
	service := req.ClientInfo.ServiceName

	name := "?"
	if req.Operation != nil {
		name = req.Operation.Name
	}

	glog.V(4).Infof("AWS request: %s %s", service, name)
}

// An interface for testing purposes.
type tokenGetter interface {
	GetAuthorizationToken(input *ecr.GetAuthorizationTokenInput) (*ecr.GetAuthorizationTokenOutput, error)
}

// The canonical implementation
type ecrTokenGetter struct {
	svc *ecr.ECR
}

func (p *ecrTokenGetter) GetAuthorizationToken(input *ecr.GetAuthorizationTokenInput) (*ecr.GetAuthorizationTokenOutput, error) {
	return p.svc.GetAuthorizationToken(input)
}

// ecrProvider is a DockerConfigProvider that gets and refreshes 12-hour tokens
// from AWS to access ECR.
type ecrProvider struct {
	getter tokenGetter
}

// Not using the package init() function: this module should be initialized only
// if using the AWS cloud provider. This way, we avoid timeouts waiting for a
// non-existent provider.
func Init() {
	credentialprovider.RegisterCredentialProvider("aws-ecr-key",
		&credentialprovider.CachingDockerConfigProvider{
			Provider: &ecrProvider{},
			// Refresh credentials a little earlier before they expire
			Lifetime: 11*time.Hour + 55*time.Minute,
		})
}

// Enabled implements DockerConfigProvider.Enabled for the AWS token-based implementation.
// For now, it gets activated only if AWS was chosen as the cloud provider.
// TODO: figure how to enable it manually for deployments that are not on AWS but still
// use ECR somehow?
func (p *ecrProvider) Enabled() bool {
	provider, err := cloudprovider.GetCloudProvider("aws", nil)
	if err != nil {
		glog.Errorf("while initializing AWS cloud provider %v", err)
		return false
	}
	if provider == nil {
		return false
	}

	zones, ok := provider.Zones()
	if !ok {
		glog.Errorf("couldn't get Zones() interface")
		return false
	}
	zone, err := zones.GetZone()
	if err != nil {
		glog.Errorf("while getting zone %v", err)
		return false
	}
	if zone.Region == "" {
		glog.Errorf("Region information is empty")
		return false
	}

	getter := &ecrTokenGetter{svc: ecr.New(session.New(&aws.Config{
		Credentials: nil,
		Region:      &zone.Region,
	}))}
	getter.svc.Handlers.Sign.PushFrontNamed(request.NamedHandler{
		Name: "k8s/logger",
		Fn:   awsHandlerLogger,
	})
	p.getter = getter

	return true
}

// Provide implements DockerConfigProvider.Provide, refreshing ECR tokens on demand
func (p *ecrProvider) Provide() credentialprovider.DockerConfig {
	cfg := credentialprovider.DockerConfig{}

	// TODO: fill in RegistryIds?
	params := &ecr.GetAuthorizationTokenInput{}
	output, err := p.getter.GetAuthorizationToken(params)
	if err != nil {
		glog.Errorf("while requesting ECR authorization token %v", err)
		return cfg
	}
	if output == nil {
		glog.Errorf("Got back no ECR token")
		return cfg
	}

	for _, data := range output.AuthorizationData {
		if data.ProxyEndpoint != nil &&
			data.AuthorizationToken != nil {
			decodedToken, err := base64.StdEncoding.DecodeString(aws.StringValue(data.AuthorizationToken))
			if err != nil {
				glog.Errorf("while decoding token for endpoint %s %v", data.ProxyEndpoint, err)
				return cfg
			}
			parts := strings.SplitN(string(decodedToken), ":", 2)
			user := parts[0]
			password := parts[1]
			entry := credentialprovider.DockerConfigEntry{
				Username: user,
				Password: password,
				// ECR doesn't care and Docker is about to obsolete it
				Email: "not@val.id",
			}

			// Add our entry for each of the supported container registry URLs
			for _, k := range registryUrls {
				cfg[k] = entry
			}
		}
	}
	return cfg
}
