package main

import (
	"io/ioutil"
	"os"

	"fmt"
	"net/http"

	"github.com/nalind/gss/pkg/gss"
	gsshttp "github.com/nalind/gss/pkg/gss/http"
	gssproxy "github.com/nalind/gss/pkg/gss/proxy/http"
)

func main() {
	var client *http.Client
	if len(os.Args) < 2 {
		fmt.Println("Usage: www-authenticate URL [gss-proxy-socket]")
		return
	}

	req, err := http.NewRequest("GET", os.Args[1], nil)
	if err != nil {
		fmt.Println(err)
		return
	}

	if len(os.Args) > 2 {
		client = &http.Client{Transport: gssproxy.NewNegotiateRoundTripper(os.Args[2], http.DefaultTransport)}
	} else {
		client = &http.Client{
			Transport: &gsshttp.NegotiateRoundTripper{
				Transport: http.DefaultTransport,
				Flags:     gss.Flags{Mutual: true},
			},
		}
	}

	resp, err := client.Do(req)
	if err != nil {
		fmt.Println(err)
		return
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		fmt.Printf("reading reponse body: %v\n", err)
	}
	fmt.Println(string(body))
}
