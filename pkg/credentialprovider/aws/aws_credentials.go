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
	"bytes"
	"encoding/base64"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/ec2metadata"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ecr"
	"k8s.io/klog"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/kubernetes/pkg/version"
)

const (
	awsChinaRegionPrefix        = "cn-"
	awsStandardDNSSuffix        = "amazonaws.com"
	awsChinaDNSSuffix           = "amazonaws.com.cn"
	registryURLTemplate         = "*.dkr.ecr.%s.%s"
	awsOsakaLocalRegion         = "ap-northeast-3"
	awsAvailabilityZoneEndpoint = "placement/availability-zone"
	awsChassisFile              = "/sys/devices/virtual/dmi/id/chassis_vendor"
	awsHypervisorFile           = "/sys/hypervisor/uuid"
	awsVersionFile              = "/sys/hypervisor/version/extra"
)

func init() {
	seen := sets.NewString()

	for _, p := range endpoints.DefaultPartitions() {
		for r := range p.Regions() {
			if !seen.Has(r) {
				registerCredentialsProvider(r)
				seen.Insert(r)
			}
		}
	}

	// ap-northeast-3 is purposely excluded from the SDK because it requires an
	// access request see: https://github.com/aws/aws-sdk-go/issues/1863
	registerCredentialsProvider(awsOsakaLocalRegion)

	// The metadata service is only available on AWS instances
	if onAWSInstance() {
		// Register the current region if we haven't seen it already just in case
		// we're in a region that's not in the SDK for some reason.
		currentRegion, err := getCurrentRegion()
		if err != nil {
			klog.Warningf("unable to detect region for current host, error: %v", err)
		} else if !seen.Has(currentRegion) {
			registerCredentialsProvider(currentRegion)
		}
	}
}

// awsHandlerLogger is a handler that logs all AWS SDK requests
// Copied from pkg/cloudprovider/providers/aws/log_handler.go
func awsHandlerLogger(req *request.Request) {
	service := req.ClientInfo.ServiceName
	region := req.Config.Region

	name := "?"
	if req.Operation != nil {
		name = req.Operation.Name
	}

	klog.V(3).Infof("AWS request: %s:%s in %s", service, name, *region)
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

// registryURL has different suffix in AWS China region
func registryURL(region string) string {
	dnsSuffix := awsStandardDNSSuffix
	// deal with aws none standard regions
	if strings.HasPrefix(region, awsChinaRegionPrefix) {
		dnsSuffix = awsChinaDNSSuffix
	}
	return fmt.Sprintf(registryURLTemplate, region, dnsSuffix)
}

// registerCredentialsProvider registers a credential provider for the specified region.
// It creates a lazy provider for each AWS region, in order to support
// cross-region ECR access. They have to be lazy because it's unlikely, but not
// impossible, that we'll use more than one.
// This should be called only if using the AWS cloud provider.
// This way, we avoid timeouts waiting for a non-existent provider.
func registerCredentialsProvider(region string) {
	klog.V(4).Infof("registering credentials provider for AWS region %q", region)

	credentialprovider.RegisterCredentialProvider("aws-ecr-"+region,
		&lazyEcrProvider{
			region:    region,
			regionURL: registryURL(region),
		})
}

func onAWSInstance() bool {
	// This test is for all modern instance types. On these instance types the
	// file will exist and contain the value "Amazon EC2\n". On older HVM
	// instances the file may exist and if it does will contain the value
	// "Xen\n". It's insufficient to differentiate the instance purely on these
	// factors because non-EC2 Xen guests will incorrectly pass the second
	// test. On older PV instances the file will not exist at all.
	//
	// This does not use the test recommended in the Amazon documentation to
	// avoid requiring root.
	//
	// See: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/identify_ec2_instances.html
	if data, err := ioutil.ReadFile(awsChassisFile); err != nil {
		klog.V(2).Infof("Error while reading %s: %v", awsChassisFile, err)
	} else if bytes.Equal(bytes.TrimSpace(data), []byte("Amazon EC2")) {
		return true
	}

	// Older PV and HVM EC2 instances will always have a prefix of ec2 on the
	// hypervisor UUID. They should generally also have an amazon suffix on the
	// hypervisor extra version.
	if data, err := ioutil.ReadFile(awsHypervisorFile); err != nil {
		klog.V(2).Infof("Error while reading %s: %v", awsHypervisorFile, err)
	} else if bytes.HasPrefix(bytes.TrimSpace(bytes.ToLower(data)), []byte("ec2")) {
		return true
	}

	// This check is an extra bit of paranoia in the off-chance that a non-EC2
	// Xen guest has a ec2 prefix on the hypervisor UUID.
	if data, err := ioutil.ReadFile(awsVersionFile); err != nil {
		klog.V(2).Infof("Error while reading %s: %v", awsVersionFile, err)
	} else if bytes.HasSuffix(bytes.TrimSpace(bytes.ToLower(data)), []byte("amazon")) {
		return true
	}

	return false
}

// Enabled implements DockerConfigProvider.Enabled for the lazy provider. It
// will return true only if we're running on a AWS instance.
func (p *lazyEcrProvider) Enabled() bool {
	return onAWSInstance()
}

// LazyProvide implements DockerConfigProvider.LazyProvide. It will be called
// by the client when attempting to pull an image and it will create the actual
// provider only when we actually need it the first time.
func (p *lazyEcrProvider) LazyProvide() *credentialprovider.DockerConfigEntry {
	if p.actualProvider == nil {
		klog.V(2).Infof("Creating ecrProvider for %s", p.region)
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
		regionURL: registryURL(region),
		getter:    getter,
	}
}

// Enabled implements DockerConfigProvider.Enabled for the AWS token-based implementation.
// For now, it gets activated only if AWS was chosen as the cloud provider.
// TODO: figure how to enable it manually for deployments that are not on AWS but still
// use ECR somehow?
func (p *ecrProvider) Enabled() bool {
	if !onAWSInstance() {
		return false
	}

	if p.region == "" {
		klog.Errorf("Called ecrProvider.Enabled() with no region set")
		return false
	}

	getter := &ecrTokenGetter{svc: ecr.New(session.New(&aws.Config{
		Credentials: nil,
		Region:      &p.region,
	}))}
	getter.svc.Handlers.Build.PushFrontNamed(request.NamedHandler{
		Name: "k8s/user-agent",
		Fn:   request.MakeAddToUserAgentHandler("kubernetes", version.Get().String()),
	})
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
		klog.Errorf("while requesting ECR authorization token %v", err)
		return cfg
	}
	if output == nil {
		klog.Errorf("Got back no ECR token")
		return cfg
	}

	for _, data := range output.AuthorizationData {
		if data.ProxyEndpoint != nil &&
			data.AuthorizationToken != nil {
			decodedToken, err := base64.StdEncoding.DecodeString(aws.StringValue(data.AuthorizationToken))
			if err != nil {
				klog.Errorf("while decoding token for endpoint %v %v", data.ProxyEndpoint, err)
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

			klog.V(3).Infof("Adding credentials for user %s in %s", user, p.region)
			// Add our config entry for this region's registry URLs
			cfg[p.regionURL] = entry

		}
	}
	return cfg
}

// getCurrentRegion queries the local EC2 metadata service and returns the name
// of the region in which the current host is running
func getCurrentRegion() (string, error) {
	// Without a timeout this hangs forever. Use a really aggressively low
	// timeout because the ec2metadata service is bound to a local interface on
	// the EC2 hosts.
	sess, err := session.NewSession(aws.NewConfig().
		WithHTTPClient(&http.Client{Timeout: 3 * time.Millisecond}).
		WithMaxRetries(1))
	if err != nil {
		return "", err
	}

	client := ec2metadata.New(sess)
	zone, err := client.GetMetadata(awsAvailabilityZoneEndpoint)
	if err != nil {
		return "", err
	}

	if v := strings.TrimSpace(zone); len(v) <= 1 {
		return "", errors.New("ec2metadata returned a blank zone")
	}

	// Zone format: ${REGION}-${LOCATION}-${NUMBER}${ZONE} (ex: us-west-2c)
	return zone[:len(zone)-1], nil
}
