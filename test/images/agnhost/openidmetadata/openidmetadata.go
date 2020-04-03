/*
Copyright 2020 The Kubernetes Authors.

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

// Package openidmetadata tests the OIDC discovery endpoints which are part of
// the ServiceAccountIssuerDiscovery feature.
package openidmetadata

import (
	"context"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"

	oidc "github.com/coreos/go-oidc"
	"github.com/spf13/cobra"
	"golang.org/x/oauth2"
	"gopkg.in/square/go-jose.v2/jwt"
	"k8s.io/client-go/rest"
)

// CmdTestServiceAccountIssuerDiscovery is used by agnhost Cobra.
var CmdTestServiceAccountIssuerDiscovery = &cobra.Command{
	Use:   "test-service-account-issuer-discovery",
	Short: "Tests the ServiceAccountIssuerDiscovery feature",
	Long: "Reads in a mounted token and attempts to verify it against the API server's " +
		"OIDC endpoints, using a third-party OIDC implementation.",
	Args: cobra.MaximumNArgs(0),
	Run:  main,
}

var (
	tokenPath          string
	audience           string
	inClusterDiscovery bool
)

func init() {
	fs := CmdTestServiceAccountIssuerDiscovery.Flags()
	fs.StringVar(&tokenPath, "token-path", "", "Path to read service account token from.")
	fs.StringVar(&audience, "audience", "", "Audience to check on received token.")
	fs.BoolVar(&inClusterDiscovery, "in-cluster-discovery", false,
		"Includes the in-cluster bearer token in request headers. "+
			"Use when validating against API server's discovery endpoints, "+
			"which require authentication.")
}

func main(cmd *cobra.Command, args []string) {
	ctx, err := withOAuth2Client(context.Background())
	if err != nil {
		log.Fatal(err)
	}

	raw, err := gettoken()
	if err != nil {
		log.Fatal(err)
	}
	log.Print("OK: Got token")
	tok, err := jwt.ParseSigned(raw)
	if err != nil {
		log.Fatal(err)
	}
	var unsafeClaims claims
	if err := tok.UnsafeClaimsWithoutVerification(&unsafeClaims); err != nil {
		log.Fatal(err)
	}
	log.Printf("OK: got issuer %s", unsafeClaims.Issuer)
	log.Printf("Full, not-validated claims: \n%#v", unsafeClaims)

	iss, err := oidc.NewProvider(ctx, unsafeClaims.Issuer)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("OK: Constructed OIDC provider for issuer %v", unsafeClaims.Issuer)

	validTok, err := iss.Verifier(&oidc.Config{ClientID: audience}).Verify(ctx, raw)
	if err != nil {
		log.Fatal(err)
	}
	log.Print("OK: Validated signature on JWT")

	var safeClaims claims
	if err := validTok.Claims(&safeClaims); err != nil {
		log.Fatal(err)
	}
	log.Print("OK: Got valid claims from token!")
	log.Printf("Full, validated claims: \n%#v", &safeClaims)
}

type kubeName struct {
	Name string `json:"name"`
	UID  string `json:"uid"`
}

type kubeClaims struct {
	Namespace      string   `json:"namespace"`
	ServiceAccount kubeName `json:"serviceaccount"`
}

type claims struct {
	jwt.Claims

	Kubernetes kubeClaims `json:"kubernetes.io"`
}

func (k *claims) String() string {
	return fmt.Sprintf("%s/%s for %s", k.Kubernetes.Namespace, k.Kubernetes.ServiceAccount.Name, k.Audience)
}

func gettoken() (string, error) {
	b, err := ioutil.ReadFile(tokenPath)
	return string(b), err
}

// withOAuth2Client returns a context that includes an HTTP Client, under the
// oauth2.HTTPClient key. If  --in-cluster-discovery is true, the client will
// use the Kubernetes InClusterConfig. Otherwise it will use
// http.DefaultTransport.
// The `oidc` library respects the oauth2.HTTPClient context key; if it is set,
// the library will use the provided http.Client rather than the default
// HTTP client.
// This allows us to ensure requests get routed to the API server for
// --in-cluster-discovery, in a client configured with the appropriate CA.
func withOAuth2Client(context.Context) (context.Context, error) {
	// TODO(mtaufen): Someday, might want to change this so that we can test
	// TokenProjection with an API audience set to the external provider with
	// requests against external endpoints (in which case we'd send
	// a different token with a non-Kubernetes audience).

	// By default, use the default http transport with the system root bundle,
	// since it's validating against the external internet.
	rt := http.DefaultTransport
	if inClusterDiscovery {
		// If in-cluster discovery, then use the in-cluster config so we can
		// authenticate with the API server.
		cfg, err := rest.InClusterConfig()
		if err != nil {
			return nil, err
		}
		rt, err = rest.TransportFor(cfg)
		if err != nil {
			return nil, fmt.Errorf("could not get roundtripper: %v", err)
		}
	}

	ctx := context.WithValue(context.Background(),
		oauth2.HTTPClient, &http.Client{
			Transport: rt,
		})
	return ctx, nil
}
