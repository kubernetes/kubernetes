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
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"time"

	"github.com/coreos/go-oidc"
	"github.com/spf13/cobra"
	"golang.org/x/oauth2"
	"gopkg.in/square/go-jose.v2/jwt"
	"k8s.io/apimachinery/pkg/util/wait"
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
	tokenPath string
	audience  string
)

func init() {
	fs := CmdTestServiceAccountIssuerDiscovery.Flags()
	fs.StringVar(&tokenPath, "token-path", "", "Path to read service account token from.")
	fs.StringVar(&audience, "audience", "", "Audience to check on received token.")
}

func main(cmd *cobra.Command, args []string) {
	raw, err := gettoken()
	if err != nil {
		log.Fatal(err)
	}
	log.Print("OK: Got token")

	/*
		  To support both in-cluster discovery and external (non kube-apiserver)
		  discovery:
		  1. Attempt with in-cluster discovery. Only trust Cluster CA.
		     If pass, exit early, successfully. This attempt includes the bearer
			 token, so we only trust the Cluster CA to avoid sending tokens to
			 some external endpoint by accident.
		  2. If in-cluster discovery doesn't pass, then try again assuming both
		     discovery doc and JWKS endpoints are external rather than being
			 served from kube-apiserver. This attempt does not pass the bearer
			 token at all.
	*/

	log.Print("validating with in-cluster discovery")
	inClusterCtx, err := withInClusterOauth2Client(context.Background())
	if err != nil {
		log.Fatal(err)
	}
	if err := validate(inClusterCtx, raw); err == nil {
		os.Exit(0)
	} else {
		log.Print("failed to validate with in-cluster discovery: ", err)
	}

	log.Print("falling back to validating with external discovery")
	externalCtx, err := withExternalOAuth2Client(context.Background())
	if err != nil {
		log.Fatal(err)
	}
	if err := validate(externalCtx, raw); err != nil {
		log.Fatal(err)
	}
}

func validate(ctx context.Context, raw string) error {
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

	if runtime.GOOS == "windows" {
		if err := ensureWindowsDNSAvailability(unsafeClaims.Issuer); err != nil {
			log.Fatal(err)
		}
	}

	iss, err := oidc.NewProvider(ctx, unsafeClaims.Issuer)
	if err != nil {
		return err
	}
	log.Printf("OK: Constructed OIDC provider for issuer %v", unsafeClaims.Issuer)

	validTok, err := iss.Verifier(&oidc.Config{
		ClientID:             audience,
		SupportedSigningAlgs: []string{oidc.RS256, oidc.ES256},
	}).Verify(ctx, raw)
	if err != nil {
		return err
	}
	log.Print("OK: Validated signature on JWT")

	var safeClaims claims
	if err := validTok.Claims(&safeClaims); err != nil {
		return err
	}
	log.Print("OK: Got valid claims from token!")
	log.Printf("Full, validated claims: \n%#v", &safeClaims)
	return nil
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
	b, err := os.ReadFile(tokenPath)
	return string(b), err
}

func withExternalOAuth2Client(ctx context.Context) (context.Context, error) {
	// Use the default http transport with the system root bundle,
	// since it's validating against the external internet.
	return context.WithValue(ctx,
		// The `oidc` library respects the oauth2.HTTPClient context key; if it is set,
		// the library will use the provided http.Client rather than the default HTTP client.
		oauth2.HTTPClient, &http.Client{
			Transport: http.DefaultTransport,
		}), nil
}

func withInClusterOauth2Client(ctx context.Context) (context.Context, error) {
	// Use the in-cluster config so we can trust and authenticate with kube-apiserver
	cfg, err := rest.InClusterConfig()
	if err != nil {
		return nil, err
	}

	rt, err := rest.TransportFor(cfg)
	if err != nil {
		return nil, fmt.Errorf("could not get roundtripper: %v", err)
	}

	return context.WithValue(ctx,
		// The `oidc` library respects the oauth2.HTTPClient context key; if it is set,
		// the library will use the provided http.Client rather than the default HTTP client.
		oauth2.HTTPClient, &http.Client{
			Transport: rt,
		}), nil
}

// DNS can be available sometime after the container starts due to the way
// networking is set up for Windows nodes with dockershim as the container runtime.
// In this case, we should make sure we are able to resolve the issuer before
// invoking oidc.NewProvider.
// See https://github.com/kubernetes/kubernetes/issues/99470 for more details.
func ensureWindowsDNSAvailability(issuer string) error {
	log.Println("Ensuring Windows DNS availability")

	u, err := url.Parse(issuer)
	if err != nil {
		return err
	}

	return wait.PollImmediate(5*time.Second, 20*time.Second, func() (bool, error) {
		ips, err := net.LookupHost(u.Host)
		if err != nil {
			log.Println(err)
			return false, nil
		}
		log.Printf("OK: Resolved host %s: %v", u.Host, ips)
		return true, nil
	})
}
