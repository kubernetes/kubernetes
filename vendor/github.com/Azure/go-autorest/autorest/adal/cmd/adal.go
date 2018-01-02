package main

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"flag"
	"fmt"
	"log"
	"strings"

	"crypto/rsa"
	"crypto/x509"
	"io/ioutil"
	"net/http"
	"os/user"

	"github.com/Azure/go-autorest/autorest/adal"
	"golang.org/x/crypto/pkcs12"
)

const (
	deviceMode       = "device"
	clientSecretMode = "secret"
	clientCertMode   = "cert"
	refreshMode      = "refresh"

	activeDirectoryEndpoint = "https://login.microsoftonline.com/"
)

type option struct {
	name  string
	value string
}

var (
	mode     string
	resource string

	tenantID      string
	applicationID string

	applicationSecret string
	certificatePath   string

	tokenCachePath string
)

func checkMandatoryOptions(mode string, options ...option) {
	for _, option := range options {
		if strings.TrimSpace(option.value) == "" {
			log.Fatalf("Authentication mode '%s' requires mandatory option '%s'.", mode, option.name)
		}
	}
}

func defaultTokenCachePath() string {
	usr, err := user.Current()
	if err != nil {
		log.Fatal(err)
	}
	defaultTokenPath := usr.HomeDir + "/.adal/accessToken.json"
	return defaultTokenPath
}

func init() {
	flag.StringVar(&mode, "mode", "device", "authentication mode (device, secret, cert, refresh)")
	flag.StringVar(&resource, "resource", "", "resource for which the token is requested")
	flag.StringVar(&tenantID, "tenantId", "", "tenant id")
	flag.StringVar(&applicationID, "applicationId", "", "application id")
	flag.StringVar(&applicationSecret, "secret", "", "application secret")
	flag.StringVar(&certificatePath, "certificatePath", "", "path to pk12/PFC application certificate")
	flag.StringVar(&tokenCachePath, "tokenCachePath", defaultTokenCachePath(), "location of oath token cache")

	flag.Parse()

	switch mode = strings.TrimSpace(mode); mode {
	case clientSecretMode:
		checkMandatoryOptions(clientSecretMode,
			option{name: "resource", value: resource},
			option{name: "tenantId", value: tenantID},
			option{name: "applicationId", value: applicationID},
			option{name: "secret", value: applicationSecret},
		)
	case clientCertMode:
		checkMandatoryOptions(clientCertMode,
			option{name: "resource", value: resource},
			option{name: "tenantId", value: tenantID},
			option{name: "applicationId", value: applicationID},
			option{name: "certificatePath", value: certificatePath},
		)
	case deviceMode:
		checkMandatoryOptions(deviceMode,
			option{name: "resource", value: resource},
			option{name: "tenantId", value: tenantID},
			option{name: "applicationId", value: applicationID},
		)
	case refreshMode:
		checkMandatoryOptions(refreshMode,
			option{name: "resource", value: resource},
			option{name: "tenantId", value: tenantID},
			option{name: "applicationId", value: applicationID},
		)
	default:
		log.Fatalln("Authentication modes 'secret, 'cert', 'device' or 'refresh' are supported.")
	}
}

func acquireTokenClientSecretFlow(oauthConfig adal.OAuthConfig,
	appliationID string,
	applicationSecret string,
	resource string,
	callbacks ...adal.TokenRefreshCallback) (*adal.ServicePrincipalToken, error) {

	spt, err := adal.NewServicePrincipalToken(
		oauthConfig,
		appliationID,
		applicationSecret,
		resource,
		callbacks...)
	if err != nil {
		return nil, err
	}

	return spt, spt.Refresh()
}

func decodePkcs12(pkcs []byte, password string) (*x509.Certificate, *rsa.PrivateKey, error) {
	privateKey, certificate, err := pkcs12.Decode(pkcs, password)
	if err != nil {
		return nil, nil, err
	}

	rsaPrivateKey, isRsaKey := privateKey.(*rsa.PrivateKey)
	if !isRsaKey {
		return nil, nil, fmt.Errorf("PKCS#12 certificate must contain an RSA private key")
	}

	return certificate, rsaPrivateKey, nil
}

func acquireTokenClientCertFlow(oauthConfig adal.OAuthConfig,
	applicationID string,
	applicationCertPath string,
	resource string,
	callbacks ...adal.TokenRefreshCallback) (*adal.ServicePrincipalToken, error) {

	certData, err := ioutil.ReadFile(certificatePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read the certificate file (%s): %v", certificatePath, err)
	}

	certificate, rsaPrivateKey, err := decodePkcs12(certData, "")
	if err != nil {
		return nil, fmt.Errorf("failed to decode pkcs12 certificate while creating spt: %v", err)
	}

	spt, err := adal.NewServicePrincipalTokenFromCertificate(
		oauthConfig,
		applicationID,
		certificate,
		rsaPrivateKey,
		resource,
		callbacks...)
	if err != nil {
		return nil, err
	}

	return spt, spt.Refresh()
}

func acquireTokenDeviceCodeFlow(oauthConfig adal.OAuthConfig,
	applicationID string,
	resource string,
	callbacks ...adal.TokenRefreshCallback) (*adal.ServicePrincipalToken, error) {

	oauthClient := &http.Client{}
	deviceCode, err := adal.InitiateDeviceAuth(
		oauthClient,
		oauthConfig,
		applicationID,
		resource)
	if err != nil {
		return nil, fmt.Errorf("Failed to start device auth flow: %s", err)
	}

	fmt.Println(*deviceCode.Message)

	token, err := adal.WaitForUserCompletion(oauthClient, deviceCode)
	if err != nil {
		return nil, fmt.Errorf("Failed to finish device auth flow: %s", err)
	}

	spt, err := adal.NewServicePrincipalTokenFromManualToken(
		oauthConfig,
		applicationID,
		resource,
		*token,
		callbacks...)
	return spt, err
}

func refreshToken(oauthConfig adal.OAuthConfig,
	applicationID string,
	resource string,
	tokenCachePath string,
	callbacks ...adal.TokenRefreshCallback) (*adal.ServicePrincipalToken, error) {

	token, err := adal.LoadToken(tokenCachePath)
	if err != nil {
		return nil, fmt.Errorf("failed to load token from cache: %v", err)
	}

	spt, err := adal.NewServicePrincipalTokenFromManualToken(
		oauthConfig,
		applicationID,
		resource,
		*token,
		callbacks...)
	if err != nil {
		return nil, err
	}
	return spt, spt.Refresh()
}

func saveToken(spt adal.Token) error {
	if tokenCachePath != "" {
		err := adal.SaveToken(tokenCachePath, 0600, spt)
		if err != nil {
			return err
		}
		log.Printf("Acquired token was saved in '%s' file\n", tokenCachePath)
		return nil

	}
	return fmt.Errorf("empty path for token cache")
}

func main() {
	oauthConfig, err := adal.NewOAuthConfig(activeDirectoryEndpoint, tenantID)
	if err != nil {
		panic(err)
	}

	callback := func(token adal.Token) error {
		return saveToken(token)
	}

	log.Printf("Authenticating with mode '%s'\n", mode)
	switch mode {
	case clientSecretMode:
		_, err = acquireTokenClientSecretFlow(
			*oauthConfig,
			applicationID,
			applicationSecret,
			resource,
			callback)
	case clientCertMode:
		_, err = acquireTokenClientCertFlow(
			*oauthConfig,
			applicationID,
			certificatePath,
			resource,
			callback)
	case deviceMode:
		var spt *adal.ServicePrincipalToken
		spt, err = acquireTokenDeviceCodeFlow(
			*oauthConfig,
			applicationID,
			resource,
			callback)
		if err == nil {
			err = saveToken(spt.Token)
		}
	case refreshMode:
		_, err = refreshToken(
			*oauthConfig,
			applicationID,
			resource,
			tokenCachePath,
			callback)
	}

	if err != nil {
		log.Fatalf("Failed to acquire a token for resource %s. Error: %v", resource, err)
	}
}
