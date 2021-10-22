// +build example

package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go/service/cloudfront/sign"
)

// Makes a request for object using CloudFront cookie signing, and outputs
// the contents of the object to stdout.
//
// Usage example:
// signCookies -file <privkey file>  -id <keyId> -r <resource pattern> -g <object to get>
func main() {
	var keyFile string  // Private key PEM file
	var keyID string    // Key pair ID of CloudFront key pair
	var resource string // CloudFront resource pattern
	var object string   // S3 object frontented by CloudFront

	flag.StringVar(&keyFile, "file", "", "private key file")
	flag.StringVar(&keyID, "id", "", "key pair id")
	flag.StringVar(&resource, "r", "", "resource to request")
	flag.StringVar(&object, "g", "", "object to get")
	flag.Parse()

	// Load the PEM file into memory so it can be used by the signer
	privKey, err := sign.LoadPEMPrivKeyFile(keyFile)
	if err != nil {
		fmt.Println("failed to load key,", err)
		return
	}

	// Create the new CookieSigner to get signed cookies for CloudFront
	// resource requests
	signer := sign.NewCookieSigner(keyID, privKey)

	// Get the cookies for the resource. These will be used
	// to make the requests with
	cookies, err := signer.Sign(resource, time.Now().Add(1*time.Hour))
	if err != nil {
		fmt.Println("failed to sign cookies", err)
		return
	}

	// Use the cookies in a http.Client to show how they allow the client
	// to request resources from CloudFront.
	req, err := http.NewRequest("GET", object, nil)
	fmt.Println("Cookies:")
	for _, c := range cookies {
		fmt.Printf("%s=%s;\n", c.Name, c.Value)
		req.AddCookie(c)
	}

	// Send and handle the response. For a successful response the object's
	// content will be written to stdout. The same process could be applied
	// to a http service written cookies to the response but using
	// http.SetCookie(w, c,) on the ResponseWriter.
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Println("failed to send request", err)
		return
	}
	defer resp.Body.Close()

	b, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		fmt.Println("failed to read requested body", err)
		return
	}

	fmt.Println("Response:", resp.Status)
	fmt.Println(string(b))
}
