package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/coreos/go-oidc/oidc"
)

func main() {
	fs := flag.NewFlagSet("oidc-example-cli", flag.ExitOnError)
	clientID := fs.String("client-id", "", "")
	clientSecret := fs.String("client-secret", "", "")
	discovery := fs.String("discovery", "https://accounts.google.com", "")

	if err := fs.Parse(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}

	if *clientID == "" {
		fmt.Println("--client-id must be set")
		os.Exit(2)
	}

	if *clientSecret == "" {
		fmt.Println("--client-secret must be set")
		os.Exit(2)
	}

	cc := oidc.ClientCredentials{
		ID:     *clientID,
		Secret: *clientSecret,
	}

	fmt.Printf("fetching provider config from %s...", *discovery)

	// NOTE: A real CLI would cache this config, or provide it via flags/config file.
	var cfg oidc.ProviderConfig
	var err error
	for {
		cfg, err = oidc.FetchProviderConfig(http.DefaultClient, *discovery)
		if err == nil {
			break
		}

		sleep := 1 * time.Second
		fmt.Printf("failed fetching provider config, trying again in %v: %v\n", sleep, err)
		time.Sleep(sleep)
	}

	fmt.Printf("fetched provider config from %s: %#v\n\n", *discovery, cfg)

	ccfg := oidc.ClientConfig{
		ProviderConfig: cfg,
		Credentials:    cc,
	}

	client, err := oidc.NewClient(ccfg)
	if err != nil {
		fmt.Printf("unable to create Client: %v\n", err)
		os.Exit(1)
	}

	tok, err := client.ClientCredsToken([]string{"openid"})
	if err != nil {
		fmt.Printf("unable to verify auth code with issuer: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("got jwt: %v\n\n", tok.Encode())

	claims, err := tok.Claims()
	if err != nil {
		fmt.Printf("unable to construct claims: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("got claims %#v...\n", claims)
}
