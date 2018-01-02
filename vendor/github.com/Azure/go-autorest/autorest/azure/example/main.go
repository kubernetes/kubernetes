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
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"strings"

	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/adal"
	"github.com/Azure/go-autorest/autorest/azure"
	"golang.org/x/crypto/pkcs12"
)

const (
	resourceGroupURLTemplate = "https://management.azure.com"
	apiVersion               = "2015-01-01"
	nativeAppClientID        = "a87032a7-203c-4bf7-913c-44c50d23409a"
	resource                 = "https://management.core.windows.net/"
)

var (
	mode           string
	tenantID       string
	subscriptionID string
	applicationID  string

	tokenCachePath string
	forceRefresh   bool
	impatient      bool

	certificatePath string
)

func init() {
	flag.StringVar(&mode, "mode", "device", "mode of operation for SPT creation")
	flag.StringVar(&certificatePath, "certificatePath", "", "path to pk12/pfx certificate")
	flag.StringVar(&applicationID, "applicationId", "", "application id")
	flag.StringVar(&tenantID, "tenantId", "", "tenant id")
	flag.StringVar(&subscriptionID, "subscriptionId", "", "subscription id")
	flag.StringVar(&tokenCachePath, "tokenCachePath", "", "location of oauth token cache")
	flag.BoolVar(&forceRefresh, "forceRefresh", false, "pass true to force a token refresh")

	flag.Parse()

	log.Printf("mode(%s) certPath(%s) appID(%s) tenantID(%s), subID(%s)\n",
		mode, certificatePath, applicationID, tenantID, subscriptionID)

	if mode == "certificate" &&
		(strings.TrimSpace(tenantID) == "" || strings.TrimSpace(subscriptionID) == "") {
		log.Fatalln("Bad usage. Using certificate mode. Please specify tenantID, subscriptionID")
	}

	if mode != "certificate" && mode != "device" {
		log.Fatalln("Bad usage. Mode must be one of 'certificate' or 'device'.")
	}

	if mode == "device" && strings.TrimSpace(applicationID) == "" {
		log.Println("Using device mode auth. Will use `azkube` clientID since none was specified on the comand line.")
		applicationID = nativeAppClientID
	}

	if mode == "certificate" && strings.TrimSpace(certificatePath) == "" {
		log.Fatalln("Bad usage. Mode 'certificate' requires the 'certificatePath' argument.")
	}

	if strings.TrimSpace(tenantID) == "" || strings.TrimSpace(subscriptionID) == "" || strings.TrimSpace(applicationID) == "" {
		log.Fatalln("Bad usage. Must specify the 'tenantId' and 'subscriptionId'")
	}
}

func getSptFromCachedToken(oauthConfig adal.OAuthConfig, clientID, resource string, callbacks ...adal.TokenRefreshCallback) (*adal.ServicePrincipalToken, error) {
	token, err := adal.LoadToken(tokenCachePath)
	if err != nil {
		return nil, fmt.Errorf("failed to load token from cache: %v", err)
	}

	spt, _ := adal.NewServicePrincipalTokenFromManualToken(
		oauthConfig,
		clientID,
		resource,
		*token,
		callbacks...)

	return spt, nil
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

func getSptFromCertificate(oauthConfig adal.OAuthConfig, clientID, resource, certicatePath string, callbacks ...adal.TokenRefreshCallback) (*adal.ServicePrincipalToken, error) {
	certData, err := ioutil.ReadFile(certificatePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read the certificate file (%s): %v", certificatePath, err)
	}

	certificate, rsaPrivateKey, err := decodePkcs12(certData, "")
	if err != nil {
		return nil, fmt.Errorf("failed to decode pkcs12 certificate while creating spt: %v", err)
	}

	spt, _ := adal.NewServicePrincipalTokenFromCertificate(
		oauthConfig,
		clientID,
		certificate,
		rsaPrivateKey,
		resource,
		callbacks...)

	return spt, nil
}

func getSptFromDeviceFlow(oauthConfig adal.OAuthConfig, clientID, resource string, callbacks ...adal.TokenRefreshCallback) (*adal.ServicePrincipalToken, error) {
	oauthClient := &autorest.Client{}
	deviceCode, err := adal.InitiateDeviceAuth(oauthClient, oauthConfig, clientID, resource)
	if err != nil {
		return nil, fmt.Errorf("failed to start device auth flow: %s", err)
	}

	fmt.Println(*deviceCode.Message)

	token, err := adal.WaitForUserCompletion(oauthClient, deviceCode)
	if err != nil {
		return nil, fmt.Errorf("failed to finish device auth flow: %s", err)
	}

	spt, err := adal.NewServicePrincipalTokenFromManualToken(
		oauthConfig,
		clientID,
		resource,
		*token,
		callbacks...)
	if err != nil {
		return nil, fmt.Errorf("failed to get oauth token from device flow: %v", err)
	}

	return spt, nil
}

func printResourceGroups(client *autorest.Client) error {
	p := map[string]interface{}{"subscription-id": subscriptionID}
	q := map[string]interface{}{"api-version": apiVersion}

	req, _ := autorest.Prepare(&http.Request{},
		autorest.AsGet(),
		autorest.WithBaseURL(resourceGroupURLTemplate),
		autorest.WithPathParameters("/subscriptions/{subscription-id}/resourcegroups", p),
		autorest.WithQueryParameters(q))

	resp, err := autorest.SendWithSender(client, req)
	if err != nil {
		return err
	}

	value := struct {
		ResourceGroups []struct {
			Name string `json:"name"`
		} `json:"value"`
	}{}

	defer resp.Body.Close()
	dec := json.NewDecoder(resp.Body)
	err = dec.Decode(&value)
	if err != nil {
		return err
	}

	var groupNames = make([]string, len(value.ResourceGroups))
	for i, name := range value.ResourceGroups {
		groupNames[i] = name.Name
	}

	log.Println("Groups:", strings.Join(groupNames, ", "))
	return err
}

func saveToken(spt adal.Token) {
	if tokenCachePath != "" {
		err := adal.SaveToken(tokenCachePath, 0600, spt)
		if err != nil {
			log.Println("error saving token", err)
		} else {
			log.Println("saved token to", tokenCachePath)
		}
	}
}

func main() {
	var spt *adal.ServicePrincipalToken
	var err error

	callback := func(t adal.Token) error {
		log.Println("refresh callback was called")
		saveToken(t)
		return nil
	}

	oauthConfig, err := adal.NewOAuthConfig(azure.PublicCloud.ActiveDirectoryEndpoint, tenantID)
	if err != nil {
		panic(err)
	}

	if tokenCachePath != "" {
		log.Println("tokenCachePath specified; attempting to load from", tokenCachePath)
		spt, err = getSptFromCachedToken(*oauthConfig, applicationID, resource, callback)
		if err != nil {
			spt = nil // just in case, this is the condition below
			log.Println("loading from cache failed:", err)
		}
	}

	if spt == nil {
		log.Println("authenticating via 'mode'", mode)
		switch mode {
		case "device":
			spt, err = getSptFromDeviceFlow(*oauthConfig, applicationID, resource, callback)
		case "certificate":
			spt, err = getSptFromCertificate(*oauthConfig, applicationID, resource, certificatePath, callback)
		}
		if err != nil {
			log.Fatalln("failed to retrieve token:", err)
		}

		// should save it as soon as you get it since Refresh won't be called for some time
		if tokenCachePath != "" {
			saveToken(spt.Token)
		}
	}

	client := &autorest.Client{}
	client.Authorizer = autorest.NewBearerAuthorizer(spt)

	printResourceGroups(client)

	if forceRefresh {
		err = spt.Refresh()
		if err != nil {
			panic(err)
		}
		printResourceGroups(client)
	}
}
