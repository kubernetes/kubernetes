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
	"errors"
	"fmt"
	"io/ioutil"
	"net/url"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ecr"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-base/version"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/credentialprovider"
)

var (
	ecrPattern = regexp.MustCompile(`^(\d{12})\.dkr\.ecr(\-fips)?\.([a-zA-Z0-9][a-zA-Z0-9-_]*)\.amazonaws\.com(\.cn)?$`)
	once       sync.Once
	isEC2      bool
)

// init registers a credential provider for each registryURLTemplate and creates
// an ECR token getter factory with a new cache to store token getters
func init() {
	credentialprovider.RegisterCredentialProvider("amazon-ecr",
		newECRProvider(&ecrTokenGetterFactory{cache: make(map[string]tokenGetter)},
			ec2ValidationImpl,
		))
}

// ecrProvider is a DockerConfigProvider that gets and refreshes tokens
// from AWS to access ECR.
type ecrProvider struct {
	cache         cache.Store
	getterFactory tokenGetterFactory
	isEC2         ec2ValidationFunc
}

var _ credentialprovider.DockerConfigProvider = &ecrProvider{}

func newECRProvider(getterFactory tokenGetterFactory, isEC2 ec2ValidationFunc) *ecrProvider {
	return &ecrProvider{
		cache:         cache.NewExpirationStore(stringKeyFunc, &ecrExpirationPolicy{}),
		getterFactory: getterFactory,
		isEC2:         isEC2,
	}
}

// Enabled implements DockerConfigProvider.Enabled.
func (p *ecrProvider) Enabled() bool {
	return true
}

type ec2ValidationFunc func() bool

// ec2ValidationImpl returns true if we detect
// an EC2 vm based on checking for the EC2 system UUID, the asset tag (for nitro
// instances), or instance credentials if the UUID is not present.
// Ref: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/identify_ec2_instances.html
func ec2ValidationImpl() bool {
	return tryValidateEC2UUID() || tryValidateEC2Creds()
}

func tryValidateEC2UUID() bool {
	hypervisor_uuid := "/sys/hypervisor/uuid"
	product_uuid := "/sys/devices/virtual/dmi/id/product_uuid"
	asset_tag := "/sys/devices/virtual/dmi/id/board_asset_tag"

	if _, err := os.Stat(hypervisor_uuid); err == nil {
		b, err := ioutil.ReadFile(hypervisor_uuid)
		if err != nil {
			klog.Errorf("error checking if this is an EC2 instance: %v", err)
		} else if strings.HasPrefix(string(b), "EC2") || strings.HasPrefix(string(b), "ec2") {
			klog.V(5).Infof("found 'ec2' in uuid %v from %v, enabling legacy AWS credential provider", string(b), hypervisor_uuid)
			return true
		}
	}

	if _, err := os.Stat(product_uuid); err == nil {
		b, err := ioutil.ReadFile(product_uuid)
		if err != nil {
			klog.Errorf("error checking if this is an EC2 instance: %v", err)
		} else if strings.HasPrefix(string(b), "EC2") || strings.HasPrefix(string(b), "ec2") {
			klog.V(5).Infof("found 'ec2' in uuid %v from %v, enabling legacy AWS credential provider", string(b), product_uuid)
			return true
		}
	}

	if _, err := os.Stat(asset_tag); err == nil {
		b, err := ioutil.ReadFile(asset_tag)
		s := strings.TrimSpace(string(b))
		if err != nil {
			klog.Errorf("error checking if this is an EC2 instance: %v", err)
		} else if strings.HasPrefix(s, "i-") && len(s) == 19 {
			// Instance ID's are 19 characters plus newline
			klog.V(5).Infof("found instance ID in %v from %v, enabling legacy AWS credential provider", string(b), asset_tag)
			return true
		}
	}
	return false
}

func tryValidateEC2Creds() bool {
	sess, err := session.NewSessionWithOptions(session.Options{
		SharedConfigState: session.SharedConfigEnable,
	})
	if err != nil {
		klog.Errorf("while validating AWS credentials %v", err)
		return false
	}
	if _, err := sess.Config.Credentials.Get(); err != nil {
		klog.Errorf("while getting AWS credentials %v", err)
		return false
	}
	klog.V(5).Infof("found aws credentials, enabling legacy AWS credential provider")
	return true
}

// Provide returns a DockerConfig with credentials from the cache if they are
// found, or from ECR
func (p *ecrProvider) Provide(image string) credentialprovider.DockerConfig {
	parsed, err := parseRepoURL(image)
	if err != nil {
		return credentialprovider.DockerConfig{}
	}

	// To prevent the AWS SDK from causing latency on non-aws platforms, only test if we are on
	// EC2 or have access to credentials once.  Attempt to do it without network calls by checking
	// for certain EC2-specific files.  Otherwise, we ask the SDK to initialize a session to see if
	// credentials are available.  On non-aws platforms, especially when a metadata endpoint is blocked,
	// this has been shown to cause 20 seconds of latency due to SDK retries
	// (see https://github.com/kubernetes/kubernetes/issues/92162)
	once.Do(func() {
		isEC2 = p.isEC2()

		if isEC2 && credentialprovider.AreLegacyCloudCredentialProvidersDisabled() {
			klog.V(4).Infof("AWS credential provider is now disabled. Please refer to sig-cloud-provider for guidance on external credential provider integration for AWS")
		}
	})

	if !isEC2 {
		return credentialprovider.DockerConfig{}
	}

	if credentialprovider.AreLegacyCloudCredentialProvidersDisabled() {
		return credentialprovider.DockerConfig{}
	}

	if cfg, exists := p.getFromCache(parsed); exists {
		klog.V(3).Infof("Got ECR credentials from cache for %s", parsed.registry)
		return cfg
	}
	klog.V(3).Info("unable to get ECR credentials from cache, checking ECR API")

	cfg, err := p.getFromECR(parsed)
	if err != nil {
		klog.Errorf("error getting credentials from ECR for %s %v", parsed.registry, err)
		return credentialprovider.DockerConfig{}
	}
	klog.V(3).Infof("Got ECR credentials from ECR API for %s", parsed.registry)
	return cfg
}

// getFromCache attempts to get credentials from the cache
func (p *ecrProvider) getFromCache(parsed *parsedURL) (credentialprovider.DockerConfig, bool) {
	cfg := credentialprovider.DockerConfig{}

	obj, exists, err := p.cache.GetByKey(parsed.registry)
	if err != nil {
		klog.Errorf("error getting ECR credentials from cache: %v", err)
		return cfg, false
	}

	if !exists {
		return cfg, false
	}

	entry := obj.(*cacheEntry)
	cfg[entry.registry] = entry.credentials
	return cfg, true
}

// getFromECR gets credentials from ECR since they are not in the cache
func (p *ecrProvider) getFromECR(parsed *parsedURL) (credentialprovider.DockerConfig, error) {
	cfg := credentialprovider.DockerConfig{}
	getter, err := p.getterFactory.GetTokenGetterForRegion(parsed.region)
	if err != nil {
		return cfg, err
	}
	params := &ecr.GetAuthorizationTokenInput{RegistryIds: []*string{aws.String(parsed.registryID)}}
	output, err := getter.GetAuthorizationToken(params)
	if err != nil {
		return cfg, err
	}
	if output == nil {
		return cfg, errors.New("authorization token is nil")
	}
	if len(output.AuthorizationData) == 0 {
		return cfg, errors.New("authorization data from response is empty")
	}
	data := output.AuthorizationData[0]
	if data.AuthorizationToken == nil {
		return cfg, errors.New("authorization token in response is nil")
	}
	entry, err := makeCacheEntry(data, parsed.registry)
	if err != nil {
		return cfg, err
	}
	if err := p.cache.Add(entry); err != nil {
		return cfg, err
	}
	cfg[entry.registry] = entry.credentials
	return cfg, nil
}

type parsedURL struct {
	registryID string
	region     string
	registry   string
}

// parseRepoURL parses and splits the registry URL into the registry ID,
// region, and registry.
// <registryID>.dkr.ecr(-fips).<region>.amazonaws.com(.cn)
func parseRepoURL(image string) (*parsedURL, error) {
	parsed, err := url.Parse("https://" + image)
	if err != nil {
		return nil, fmt.Errorf("error parsing image %s %v", image, err)
	}
	splitURL := ecrPattern.FindStringSubmatch(parsed.Hostname())
	if len(splitURL) == 0 {
		return nil, fmt.Errorf("%s is not a valid ECR repository URL", parsed.Hostname())
	}
	return &parsedURL{
		registryID: splitURL[1],
		region:     splitURL[3],
		registry:   parsed.Hostname(),
	}, nil
}

// tokenGetter is for testing purposes
type tokenGetter interface {
	GetAuthorizationToken(input *ecr.GetAuthorizationTokenInput) (*ecr.GetAuthorizationTokenOutput, error)
}

// tokenGetterFactory is for testing purposes
type tokenGetterFactory interface {
	GetTokenGetterForRegion(string) (tokenGetter, error)
}

// ecrTokenGetterFactory stores a token getter per region
type ecrTokenGetterFactory struct {
	cache map[string]tokenGetter
	mutex sync.Mutex
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

func newECRTokenGetter(region string) (tokenGetter, error) {
	sess, err := session.NewSessionWithOptions(session.Options{
		Config:            aws.Config{Region: aws.String(region)},
		SharedConfigState: session.SharedConfigEnable,
	})
	if err != nil {
		return nil, err
	}
	getter := &ecrTokenGetter{svc: ecr.New(sess)}
	getter.svc.Handlers.Build.PushFrontNamed(request.NamedHandler{
		Name: "k8s/user-agent",
		Fn:   request.MakeAddToUserAgentHandler("kubernetes", version.Get().String()),
	})
	getter.svc.Handlers.Sign.PushFrontNamed(request.NamedHandler{
		Name: "k8s/logger",
		Fn:   awsHandlerLogger,
	})
	return getter, nil
}

// GetTokenGetterForRegion gets the token getter for the requested region. If it
// doesn't exist, it creates a new ECR token getter
func (f *ecrTokenGetterFactory) GetTokenGetterForRegion(region string) (tokenGetter, error) {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	if getter, ok := f.cache[region]; ok {
		return getter, nil
	}
	getter, err := newECRTokenGetter(region)
	if err != nil {
		return nil, fmt.Errorf("unable to create token getter for region %v %v", region, err)
	}
	f.cache[region] = getter
	return getter, nil
}

// The canonical implementation
type ecrTokenGetter struct {
	svc *ecr.ECR
}

// GetAuthorizationToken gets the ECR authorization token using the ECR API
func (p *ecrTokenGetter) GetAuthorizationToken(input *ecr.GetAuthorizationTokenInput) (*ecr.GetAuthorizationTokenOutput, error) {
	return p.svc.GetAuthorizationToken(input)
}

type cacheEntry struct {
	expiresAt   time.Time
	credentials credentialprovider.DockerConfigEntry
	registry    string
}

// makeCacheEntry decodes the ECR authorization entry and re-packages it into a
// cacheEntry.
func makeCacheEntry(data *ecr.AuthorizationData, registry string) (*cacheEntry, error) {
	decodedToken, err := base64.StdEncoding.DecodeString(aws.StringValue(data.AuthorizationToken))
	if err != nil {
		return nil, fmt.Errorf("error decoding ECR authorization token: %v", err)
	}
	parts := strings.SplitN(string(decodedToken), ":", 2)
	if len(parts) < 2 {
		return nil, errors.New("error getting username and password from authorization token")
	}
	creds := credentialprovider.DockerConfigEntry{
		Username: parts[0],
		Password: parts[1],
		Email:    "not@val.id", // ECR doesn't care and Docker is about to obsolete it
	}
	if data.ExpiresAt == nil {
		return nil, errors.New("authorization data expiresAt is nil")
	}
	return &cacheEntry{
		expiresAt:   data.ExpiresAt.Add(-1 * wait.Jitter(30*time.Minute, 0.2)),
		credentials: creds,
		registry:    registry,
	}, nil
}

// ecrExpirationPolicy implements ExpirationPolicy from client-go.
type ecrExpirationPolicy struct{}

// stringKeyFunc returns the cache key as a string
func stringKeyFunc(obj interface{}) (string, error) {
	key := obj.(*cacheEntry).registry
	return key, nil
}

// IsExpired checks if the ECR credentials are expired.
func (p *ecrExpirationPolicy) IsExpired(entry *cache.TimestampedEntry) bool {
	return time.Now().After(entry.Obj.(*cacheEntry).expiresAt)
}
