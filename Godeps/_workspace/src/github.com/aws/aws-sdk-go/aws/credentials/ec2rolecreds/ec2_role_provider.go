package ec2rolecreds

import (
	"bufio"
	"encoding/json"
	"fmt"
	"path"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/ec2metadata"
)

// A EC2RoleProvider retrieves credentials from the EC2 service, and keeps track if
// those credentials are expired.
//
// Example how to configure the EC2RoleProvider with custom http Client, Endpoint
// or ExpiryWindow
//
//     p := &ec2rolecreds.EC2RoleProvider{
//         // Pass in a custom timeout to be used when requesting
//         // IAM EC2 Role credentials.
//         Client: &http.Client{
//             Timeout: 10 * time.Second,
//         },
//         // Use default EC2 Role metadata endpoint, Alternate endpoints can be
//         // specified setting Endpoint to something else.
//         Endpoint: "",
//         // Do not use early expiry of credentials. If a non zero value is
//         // specified the credentials will be expired early
//         ExpiryWindow: 0,
//     }
type EC2RoleProvider struct {
	credentials.Expiry

	// EC2Metadata client to use when connecting to EC2 metadata service
	Client *ec2metadata.Client

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

// NewCredentials returns a pointer to a new Credentials object
// wrapping the EC2RoleProvider.
//
// Takes a custom http.Client which can be configured for custom handling of
// things such as timeout.
//
// Endpoint is the URL that the EC2RoleProvider will connect to when retrieving
// role and credentials.
//
// Window is the expiry window that will be subtracted from the expiry returned
// by the role credential request. This is done so that the credentials will
// expire sooner than their actual lifespan.
func NewCredentials(client *ec2metadata.Client, window time.Duration) *credentials.Credentials {
	return credentials.NewCredentials(&EC2RoleProvider{
		Client:       client,
		ExpiryWindow: window,
	})
}

// Retrieve retrieves credentials from the EC2 service.
// Error will be returned if the request fails, or unable to extract
// the desired credentials.
func (m *EC2RoleProvider) Retrieve() (credentials.Value, error) {
	if m.Client == nil {
		m.Client = ec2metadata.New(nil)
	}

	credsList, err := requestCredList(m.Client)
	if err != nil {
		return credentials.Value{}, err
	}

	if len(credsList) == 0 {
		return credentials.Value{}, awserr.New("EmptyEC2RoleList", "empty EC2 Role list", nil)
	}
	credsName := credsList[0]

	roleCreds, err := requestCred(m.Client, credsName)
	if err != nil {
		return credentials.Value{}, err
	}

	m.SetExpiration(roleCreds.Expiration, m.ExpiryWindow)

	return credentials.Value{
		AccessKeyID:     roleCreds.AccessKeyID,
		SecretAccessKey: roleCreds.SecretAccessKey,
		SessionToken:    roleCreds.Token,
	}, nil
}

// A ec2RoleCredRespBody provides the shape for deserializing credential
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

const iamSecurityCredsPath = "/iam/security-credentials"

// requestCredList requests a list of credentials from the EC2 service.
// If there are no credentials, or there is an error making or receiving the request
func requestCredList(client *ec2metadata.Client) ([]string, error) {
	resp, err := client.GetMetadata(iamSecurityCredsPath)
	if err != nil {
		return nil, awserr.New("EC2RoleRequestError", "failed to list EC2 Roles", err)
	}

	credsList := []string{}
	s := bufio.NewScanner(strings.NewReader(resp))
	for s.Scan() {
		credsList = append(credsList, s.Text())
	}

	if err := s.Err(); err != nil {
		return nil, awserr.New("SerializationError", "failed to read list of EC2 Roles", err)
	}

	return credsList, nil
}

// requestCred requests the credentials for a specific credentials from the EC2 service.
//
// If the credentials cannot be found, or there is an error reading the response
// and error will be returned.
func requestCred(client *ec2metadata.Client, credsName string) (ec2RoleCredRespBody, error) {
	resp, err := client.GetMetadata(path.Join(iamSecurityCredsPath, credsName))
	if err != nil {
		return ec2RoleCredRespBody{},
			awserr.New("EC2RoleRequestError",
				fmt.Sprintf("failed to get %s EC2 Role credentials", credsName),
				err)
	}

	respCreds := ec2RoleCredRespBody{}
	if err := json.NewDecoder(strings.NewReader(resp)).Decode(&respCreds); err != nil {
		return ec2RoleCredRespBody{},
			awserr.New("SerializationError",
				fmt.Sprintf("failed to decode %s EC2 Role credentials", credsName),
				err)
	}

	if respCreds.Code != "Success" {
		// If an error code was returned something failed requesting the role.
		return ec2RoleCredRespBody{}, awserr.New(respCreds.Code, respCreds.Message, nil)
	}

	return respCreds, nil
}
