package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"time"

	"github.com/coreos/go-oidc/oidc"
)

var (
	pathCallback      = "/oauth2callback"
	defaultListenHost = "127.0.0.1:5555"
)

func main() {
	log.SetOutput(os.Stderr)

	fs := flag.NewFlagSet("oidc-example-app", flag.ExitOnError)
	listen := fs.String("listen", defaultListenHost, "serve traffic on this address (<host>:<port>)")
	redirectURL := fs.String("redirect-url", fmt.Sprintf("http://%s%s", defaultListenHost, pathCallback), "")
	clientID := fs.String("client-id", "", "")
	clientSecret := fs.String("client-secret", "", "")
	discovery := fs.String("discovery", "https://accounts.google.com", "")

	if err := fs.Parse(os.Args[1:]); err != nil {
		log.Fatalf("failed parsing flags: %v", err)
	}

	if *clientID == "" {
		log.Fatal("--client-id must be set")
	}

	if *clientSecret == "" {
		log.Fatal("--client-secret must be set")
	}

	_, _, err := net.SplitHostPort(*listen)
	if err != nil {
		log.Fatalf("unable to parse host:port from --listen flag: %v", err)
	}

	cc := oidc.ClientCredentials{
		ID:     *clientID,
		Secret: *clientSecret,
	}

	log.Printf("fetching provider config from %s...", *discovery)

	var cfg oidc.ProviderConfig
	for {
		cfg, err = oidc.FetchProviderConfig(http.DefaultClient, *discovery)
		if err == nil {
			break
		}

		sleep := 3 * time.Second
		log.Printf("failed fetching provider config, trying again in %v: %v", sleep, err)
		time.Sleep(sleep)
	}

	log.Printf("fetched provider config from %s: %#v", *discovery, cfg)

	ccfg := oidc.ClientConfig{
		ProviderConfig: cfg,
		Credentials:    cc,
		RedirectURL:    *redirectURL,
	}

	client, err := oidc.NewClient(ccfg)
	if err != nil {
		log.Fatalf("unable to create Client: %v", err)
	}

	client.SyncProviderConfig(*discovery)

	redirectURLParsed, err := url.Parse(*redirectURL)
	if err != nil {
		log.Fatalf("unable to parse url from --redirect-url flag: %v", err)
	}
	hdlr := NewClientHandler(client, *redirectURLParsed)
	httpsrv := &http.Server{
		Addr:    fmt.Sprintf(*listen),
		Handler: hdlr,
	}

	log.Printf("binding to %s...", httpsrv.Addr)
	log.Fatal(httpsrv.ListenAndServe())
}

func NewClientHandler(c *oidc.Client, cbURL url.URL) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", handleIndex)
	mux.HandleFunc("/login", handleLoginFunc(c))
	mux.HandleFunc(pathCallback, handleCallbackFunc(c))
	return mux
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	w.Write([]byte("<a href='/login'>login</a>"))
}

func handleLoginFunc(c *oidc.Client) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		oac, err := c.OAuthClient()
		if err != nil {
			panic("unable to proceed")
		}

		u, err := url.Parse(oac.AuthCodeURL("", "", ""))
		if err != nil {
			panic("unable to proceed")
		}
		http.Redirect(w, r, u.String(), http.StatusFound)
	}
}

func handleCallbackFunc(c *oidc.Client) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		code := r.URL.Query().Get("code")
		if code == "" {
			writeError(w, http.StatusBadRequest, "code query param must be set")
			return
		}

		tok, err := c.ExchangeAuthCode(code)
		if err != nil {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("unable to verify auth code with issuer: %v", err))
			return
		}

		claims, err := tok.Claims()
		if err != nil {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("unable to construct claims: %v", err))
			return
		}

		s := fmt.Sprintf("claims: %v", claims)
		w.Write([]byte(s))
	}
}

func writeError(w http.ResponseWriter, code int, msg string) {
	e := struct {
		Error string `json:"error"`
	}{
		Error: msg,
	}
	b, err := json.Marshal(e)
	if err != nil {
		log.Printf("Failed marshaling %#v to JSON: %v", e, err)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	w.Write(b)
}
