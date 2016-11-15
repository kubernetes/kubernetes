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

package credentials

import (
	"encoding/base64"
	"fmt"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ecr"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/credentialprovider"
)

// AWSRegions is the complete list of regions known to the AWS cloudprovider
// and credentialprovider.
var AWSRegions = [...]string{
	"us-east-1",
	"us-east-2",
	"us-west-1",
	"us-west-2",
	"eu-west-1",
	"eu-central-1",
	"ap-south-1",
	"ap-southeast-1",
	"ap-southeast-2",
	"ap-northeast-1",
	"ap-northeast-2",
	"cn-north-1",
	"us-gov-west-1",
	"sa-east-1",
}

const registryURLTemplate = "*.dkr.ecr.%s.amazonaws.com"

// awsHandlerLogger is a handler that logs all AWS SDK requests
// Copied from pkg/cloudprovider/providers/aws/log_handler.go
func awsHandlerLogger(req *request.Request) {
	service := req.ClientInfo.ServiceName
	region := req.Config.Region

	name := "?"
	if req.Operation != nil {
		name = req.Operation.Name
	}

	glog.V(3).Infof("AWS request: %s:%s in %s", service, name, *region)
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

// lazyEcrProvider is a DockerConfigProvider that creates on demand an
// ecrProvider for a given region and then proxies requests to it.
type lazyEcrProvider struct {
	region         string
	regionURL      string
	actualProvider *credentialprovider.CachingDockerConfigProvider
}

var _ credentialprovider.DockerConfigProvider = &lazyEcrProvider{}

// ecrProvider is a DockerConfigProvider that gets and refreshes 12-hour tokens
// from AWS to access ECR.
type ecrProvider struct {
	region    string
	regionURL string
	getter    tokenGetter
}

var _ credentialprovider.DockerConfigProvider = &ecrProvider{}

// Init creates a lazy provider for each AWS region, in order to support
// cross-region ECR access. They have to be lazy because it's unlikely, but not
// impossible, that we'll use more than one.
// Not using the package init() function: this module should be initialized only
// if using the AWS cloud provider. This way, we avoid timeouts waiting for a
// non-existent provider.
func Init() {
	for _, region := range AWSRegions {
		credentialprovider.RegisterCredentialProvider("aws-ecr-"+region,
			&lazyEcrProvider{
				region:    region,
				regionURL: fmt.Sprintf(registryURLTemplate, region),
			})
	}

}

// Enabled implements DockerConfigProvider.Enabled for the lazy provider.
// Since we perform no checks/work of our own and actualProvider is only created
// later at image pulling time (if ever), always return true.
func (p *lazyEcrProvider) Enabled() bool {
	return true
}

// LazyProvide implements DockerConfigProvider.LazyProvide. It will be called
// by the client when attempting to pull an image and it will create the actual
// provider only when we actually need it the first time.
func (p *lazyEcrProvider) LazyProvide() *credentialprovider.DockerConfigEntry {
	if p.actualProvider == nil {
		glog.V(2).Infof("Creating ecrProvider for %s", p.region)
		p.actualProvider = &credentialprovider.CachingDockerConfigProvider{
			Provider: newEcrProvider(p.region, nil),
			// Refresh credentials a little earlier than expiration time
			Lifetime: 11*time.Hour + 55*time.Minute,
		}
		if !p.actualProvider.Enabled() {
			return nil
		}
	}
	entry := p.actualProvider.Provide()[p.regionURL]
	return &entry
}

// Provide implements DockerConfigProvider.Provide, creating dummy credentials.
// Client code will call Provider.LazyProvide() at image pulling time.
func (p *lazyEcrProvider) Provide() credentialprovider.DockerConfig {
	entry := credentialprovider.DockerConfigEntry{
		Provider: p,
	}
	cfg := credentialprovider.DockerConfig{}
	cfg[p.regionURL] = entry
	return cfg
}

func newEcrProvider(region string, getter tokenGetter) *ecrProvider {
	return &ecrProvider{
		region:    region,
		regionURL: fmt.Sprintf(registryURLTemplate, region),
		getter:    getter,
	}
}

// Enabled implements DockerConfigProvider.Enabled for the AWS token-based implementation.
// For now, it gets activated only if AWS was chosen as the cloud provider.
// TODO: figure how to enable it manually for deployments that are not on AWS but still
// use ECR somehow?
func (p *ecrProvider) Enabled() bool {
	if p.region == "" {
		glog.Errorf("Called ecrProvider.Enabled() with no region set")
		return false
	}

	getter := &ecrTokenGetter{svc: ecr.New(session.New(&aws.Config{
		Credentials: nil,
		Region:      &p.region,
	}))}
	getter.svc.Handlers.Sign.PushFrontNamed(request.NamedHandler{
		Name: "k8s/logger",
		Fn:   awsHandlerLogger,
	})
	p.getter = getter

	return true
}

// LazyProvide implements DockerConfigProvider.LazyProvide. Should never be called.
func (p *ecrProvider) LazyProvide() *credentialprovider.DockerConfigEntry {
	return nil
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
				glog.Errorf("while decoding token for endpoint %v %v", data.ProxyEndpoint, err)
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

			glog.V(3).Infof("Adding credentials for user %s in %s", user, p.region)
			// Add our config entry for this region's registry URLs
			cfg[p.regionURL] = entry

		}
	}
	return cfg
}
