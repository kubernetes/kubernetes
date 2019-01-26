package ec2rolecreds

import (
	"bufio"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/ec2metadata"
	"github.com/aws/aws-sdk-go/internal/sdkuri"
)

// ProviderName provides a name of EC2Role provider
const ProviderName = "EC2RoleProvider"

// A EC2RoleProvider retrieves credentials from the EC2 service, and keeps track if
// those credentials are expired.
//
// Example how to configure the EC2RoleProvider with custom http Client, Endpoint
// or ExpiryWindow
//
//     p := &ec2rolecreds.EC2RoleProvider{
//         // Pass in a custom timeout to be used when requesting
//         // IAM EC2 Role credentials.
//         Client: ec2metadata.New(sess, aws.Config{
//             HTTPClient: &http.Client{Timeout: 10 * time.Second},
//         }),
//
//         // Do not use early expiry of credentials. If a non zero value is
//         // specified the credentials will be expired early
//         ExpiryWindow: 0,
//     }
type EC2RoleProvider struct {
	credentials.Expiry

	// Required EC2Metadata client to use when connecting to EC2 metadata service.
	Client *ec2metadata.EC2Metadata

	// ExpiryWindow will allow the credentials to trigger refreshing prior to
	// the credentials actually expiring. This is beneficial so race conditions
	// with expiring credentials do not cause request to fail unexpectedly
	// due to ExpiredTokenException exceptions.
	//
	// So a ExpiryWindow of 10s would cause calls to IsExpired() to return true
	// 10 seconds before the credentials are actually expired.
	//
	// If ExpiryWindow is 0 or less it will be ignored.
	ExpiryWindow time.Duration
}

// NewCredentials returns a pointer to a new Credentials object wrapping
// the EC2RoleProvider. Takes a ConfigProvider to create a EC2Metadata client.
// The ConfigProvider is satisfied by the session.Session type.
func NewCredentials(c client.ConfigProvider, options ...func(*EC2RoleProvider)) *credentials.Credentials {
	p := &EC2RoleProvider{
		Client: ec2metadata.New(c),
	}

	for _, option := range options {
		option(p)
	}

	return credentials.NewCredentials(p)
}

// NewCredentialsWithClient returns a pointer to a new Credentials object wrapping
// the EC2RoleProvider. Takes a EC2Metadata client to use when connecting to EC2
// metadata service.
func NewCredentialsWithClient(client *ec2metadata.EC2Metadata, options ...func(*EC2RoleProvider)) *credentials.Credentials {
	p := &EC2RoleProvider{
		Client: client,
	}

	for _, option := range options {
		option(p)
	}

	return credentials.NewCredentials(p)
}

// Retrieve retrieves credentials from the EC2 service.
// Error will be returned if the request fails, or unable to extract
// the desired credentials.
func (m *EC2RoleProvider) Retrieve() (credentials.Value, error) {
	credsList, err := requestCredList(m.Client)
	if err != nil {
		return credentials.Value{ProviderName: ProviderName}, err
	}

	if len(credsList) == 0 {
		return credentials.Value{ProviderName: ProviderName}, awserr.New("EmptyEC2RoleList", "empty EC2 Role list", nil)
	}
	credsName := credsList[0]

	roleCreds, err := requestCred(m.Client, credsName)
	if err != nil {
		return credentials.Value{ProviderName: ProviderName}, err
	}

	m.SetExpiration(roleCreds.Expiration, m.ExpiryWindow)

	return credentials.Value{
		AccessKeyID:     roleCreds.AccessKeyID,
		SecretAccessKey: roleCreds.SecretAccessKey,
		SessionToken:    roleCreds.Token,
		ProviderName:    ProviderName,
	}, nil
}

// A ec2RoleCredRespBody provides the shape for unmarshaling credential
// request responses.
type ec2RoleCredRespBody struct {
	// Success State
	Expiration      time.Time
	AccessKeyID     string
	SecretAccessKey string
	Token           string

	// Error state
	Code    string
	Message string
}

const iamSecurityCredsPath = "iam/security-credentials/"

// requestCredList requests a list of credentials from the EC2 service.
// If there are no credentials, or there is an error making or receiving the request
func requestCredList(client *ec2metadata.EC2Metadata) ([]string, error) {
	resp, err := client.GetMetadata(iamSecurityCredsPath)
	if err != nil {
		return nil, awserr.New("EC2RoleRequestError", "no EC2 instance role found", err)
	}

	credsList := []string{}
	s := bufio.NewScanner(strings.NewReader(resp))
	for s.Scan() {
		credsList = append(credsList, s.Text())
	}

	if err := s.Err(); err != nil {
		return nil, awserr.New("SerializationError", "failed to read EC2 instance role from metadata service", err)
	}

	return credsList, nil
}

// requestCred requests the credentials for a specific credentials from the EC2 service.
//
// If the credentials cannot be found, or there is an error reading the response
// and error will be returned.
func requestCred(client *ec2metadata.EC2Metadata, credsName string) (ec2RoleCredRespBody, error) {
	resp, err := client.GetMetadata(sdkuri.PathJoin(iamSecurityCredsPath, credsName))
	if err != nil {
		return ec2RoleCredRespBody{},
			awserr.New("EC2RoleRequestError",
				fmt.Sprintf("failed to get %s EC2 instance role credentials", credsName),
				err)
	}

	respCreds := ec2RoleCredRespBody{}
	if err := json.NewDecoder(strings.NewReader(resp)).Decode(&respCreds); err != nil {
		return ec2RoleCredRespBody{},
			awserr.New("SerializationError",
				fmt.Sprintf("failed to decode %s EC2 instance role credentials", credsName),
				err)
	}

	if respCreds.Code != "Success" {
		// If an error code was returned something failed requesting the role.
		return ec2RoleCredRespBody{}, awserr.New(respCreds.Code, respCreds.Message, nil)
	}

	return respCreds, nil
}
