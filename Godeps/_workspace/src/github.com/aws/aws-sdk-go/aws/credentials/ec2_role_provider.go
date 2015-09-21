package credentials

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go/internal/apierr"
)

const metadataCredentialsEndpoint = "http://169.254.169.254/latest/meta-data/iam/security-credentials/"

// A EC2RoleProvider retrieves credentials from the EC2 service, and keeps track if
// those credentials are expired.
//
// Example how to configure the EC2RoleProvider with custom http Client, Endpoint
// or ExpiryWindow
//
//     p := &credentials.EC2RoleProvider{
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
//
type EC2RoleProvider struct {
	// Endpoint must be fully quantified URL
	Endpoint string

	// HTTP client to use when connecting to EC2 service
	Client *http.Client

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

	// The date/time at which the credentials expire.
	expiresOn time.Time
}

// NewEC2RoleCredentials returns a pointer to a new Credentials object
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
func NewEC2RoleCredentials(client *http.Client, endpoint string, window time.Duration) *Credentials {
	return NewCredentials(&EC2RoleProvider{
		Endpoint:     endpoint,
		Client:       client,
		ExpiryWindow: window,
	})
}

// Retrieve retrieves credentials from the EC2 service.
// Error will be returned if the request fails, or unable to extract
// the desired credentials.
func (m *EC2RoleProvider) Retrieve() (Value, error) {
	if m.Client == nil {
		m.Client = http.DefaultClient
	}
	if m.Endpoint == "" {
		m.Endpoint = metadataCredentialsEndpoint
	}

	credsList, err := requestCredList(m.Client, m.Endpoint)
	if err != nil {
		return Value{}, err
	}

	if len(credsList) == 0 {
		return Value{}, apierr.New("EmptyEC2RoleList", "empty EC2 Role list", nil)
	}
	credsName := credsList[0]

	roleCreds, err := requestCred(m.Client, m.Endpoint, credsName)
	if err != nil {
		return Value{}, err
	}

	m.expiresOn = roleCreds.Expiration
	if m.ExpiryWindow > 0 {
		// Offset based on expiry window if set.
		m.expiresOn = m.expiresOn.Add(-m.ExpiryWindow)
	}

	return Value{
		AccessKeyID:     roleCreds.AccessKeyID,
		SecretAccessKey: roleCreds.SecretAccessKey,
		SessionToken:    roleCreds.Token,
	}, nil
}

// IsExpired returns if the credentials are expired.
func (m *EC2RoleProvider) IsExpired() bool {
	return m.expiresOn.Before(currentTime())
}

// A ec2RoleCredRespBody provides the shape for deserializing credential
// request responses.
type ec2RoleCredRespBody struct {
	Expiration      time.Time
	AccessKeyID     string
	SecretAccessKey string
	Token           string
}

// requestCredList requests a list of credentials from the EC2 service.
// If there are no credentials, or there is an error making or receiving the request
func requestCredList(client *http.Client, endpoint string) ([]string, error) {
	resp, err := client.Get(endpoint)
	if err != nil {
		return nil, apierr.New("ListEC2Role", "failed to list EC2 Roles", err)
	}
	defer resp.Body.Close()

	credsList := []string{}
	s := bufio.NewScanner(resp.Body)
	for s.Scan() {
		credsList = append(credsList, s.Text())
	}

	if err := s.Err(); err != nil {
		return nil, apierr.New("ReadEC2Role", "failed to read list of EC2 Roles", err)
	}

	return credsList, nil
}

// requestCred requests the credentials for a specific credentials from the EC2 service.
//
// If the credentials cannot be found, or there is an error reading the response
// and error will be returned.
func requestCred(client *http.Client, endpoint, credsName string) (*ec2RoleCredRespBody, error) {
	resp, err := client.Get(endpoint + credsName)
	if err != nil {
		return nil, apierr.New("GetEC2RoleCredentials",
			fmt.Sprintf("failed to get %s EC2 Role credentials", credsName),
			err)
	}
	defer resp.Body.Close()

	respCreds := &ec2RoleCredRespBody{}
	if err := json.NewDecoder(resp.Body).Decode(respCreds); err != nil {
		return nil, apierr.New("DecodeEC2RoleCredentials",
			fmt.Sprintf("failed to decode %s EC2 Role credentials", credsName),
			err)
	}

	return respCreds, nil
}
