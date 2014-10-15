// +build acceptance,old

package main

import (
	"crypto/rand"
	"fmt"
	"github.com/rackspace/gophercloud"
	"os"
	"strings"
	"time"
)

// getCredentials will verify existence of needed credential information
// provided through environment variables.  This function will not return
// if at least one piece of required information is missing.
func getCredentials() (provider, username, password, apiKey string) {
	provider = os.Getenv("SDK_PROVIDER")
	username = os.Getenv("SDK_USERNAME")
	password = os.Getenv("SDK_PASSWORD")
	apiKey = os.Getenv("SDK_API_KEY")
	var authURL = os.Getenv("OS_AUTH_URL")

	if (provider == "") || (username == "") || (password == "") {
		fmt.Fprintf(os.Stderr, "One or more of the following environment variables aren't set:\n")
		fmt.Fprintf(os.Stderr, "  SDK_PROVIDER=\"%s\"\n", provider)
		fmt.Fprintf(os.Stderr, "  SDK_USERNAME=\"%s\"\n", username)
		fmt.Fprintf(os.Stderr, "  SDK_PASSWORD=\"%s\"\n", password)
		os.Exit(1)
	}

	if strings.Contains(provider, "rackspace") && (authURL != "") {
		provider = authURL + "/v2.0/tokens"
	}

	return
}

// randomString generates a string of given length, but random content.
// All content will be within the ASCII graphic character set.
// (Implementation from Even Shaw's contribution on
// http://stackoverflow.com/questions/12771930/what-is-the-fastest-way-to-generate-a-long-random-string-in-go).
func randomString(prefix string, n int) string {
	const alphanum = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	var bytes = make([]byte, n)
	rand.Read(bytes)
	for i, b := range bytes {
		bytes[i] = alphanum[b%byte(len(alphanum))]
	}
	return prefix + string(bytes)
}

// aSuitableImage finds a minimal image for use in dynamically creating servers.
// If none can be found, this function will panic.
func aSuitableImage(api gophercloud.CloudServersProvider) string {
	images, err := api.ListImages()
	if err != nil {
		panic(err)
	}

	// TODO(sfalvo):
	// Works for Rackspace, might not work for your provider!
	// Need to figure out why ListImages() provides 0 values for
	// Ram and Disk fields.
	//
	// Until then, just return Ubuntu 12.04 LTS.
	for i := 0; i < len(images); i++ {
		if strings.Contains(images[i].Name, "Ubuntu 12.04 LTS") {
			return images[i].Id
		}
	}
	panic("Image for Ubuntu 12.04 LTS not found.")
}

// aSuitableFlavor finds the minimum flavor capable of running the test image
// chosen by aSuitableImage.  If none can be found, this function will panic.
func aSuitableFlavor(api gophercloud.CloudServersProvider) string {
	flavors, err := api.ListFlavors()
	if err != nil {
		panic(err)
	}

	// TODO(sfalvo):
	// Works for Rackspace, might not work for your provider!
	// Need to figure out why ListFlavors() provides 0 values for
	// Ram and Disk fields.
	//
	// Until then, just return Ubuntu 12.04 LTS.
	for i := 0; i < len(flavors); i++ {
		if flavors[i].Id == "2" {
			return flavors[i].Id
		}
	}
	panic("Flavor 2 (512MB 1-core 20GB machine) not found.")
}

// createServer creates a new server in a manner compatible with acceptance testing.
// In particular, it ensures that the name of the server always starts with "ACPTTEST--",
// which the delete servers acceptance test relies on to identify servers to delete.
// Passing in empty image and flavor references will force the use of reasonable defaults.
// An empty name string will result in a dynamically created name prefixed with "ACPTTEST--".
// A blank admin password will cause a password to be automatically generated; however,
// at present no means of recovering this password exists, as no acceptance tests yet require
// this data.
func createServer(servers gophercloud.CloudServersProvider, imageRef, flavorRef, name, adminPass string) (string, error) {
	if imageRef == "" {
		imageRef = aSuitableImage(servers)
	}

	if flavorRef == "" {
		flavorRef = aSuitableFlavor(servers)
	}

	if len(name) < 1 {
		name = randomString("ACPTTEST", 16)
	}

	if (len(name) < 8) || (name[0:8] != "ACPTTEST") {
		name = fmt.Sprintf("ACPTTEST--%s", name)
	}

	newServer, err := servers.CreateServer(gophercloud.NewServer{
		Name:      name,
		ImageRef:  imageRef,
		FlavorRef: flavorRef,
		AdminPass: adminPass,
	})

	if err != nil {
		return "", err
	}

	return newServer.Id, nil
}

// findAlternativeFlavor locates a flavor to resize a server to.  It is guaranteed to be different
// than what aSuitableFlavor() returns.  If none could be found, this function will panic.
func findAlternativeFlavor() string {
	return "3" // 1GB image, up from 512MB image
}

// findAlternativeImage locates an image to resize or rebuild a server with.  It is guaranteed to be
// different than what aSuitableImage() returns.  If none could be found, this function will panic.
func findAlternativeImage() string {
	return "c6f9c411-e708-4952-91e5-62ded5ea4d3e"
}

// withIdentity authenticates the user against the provider's identity service, and provides an
// accessor for additional services.
func withIdentity(ar bool, f func(gophercloud.AccessProvider)) {
	_, _, _, apiKey := getCredentials()
	if len(apiKey) == 0 {
		withPasswordIdentity(ar, f)
	} else {
		withAPIKeyIdentity(ar, f)
	}
}

func withPasswordIdentity(ar bool, f func(gophercloud.AccessProvider)) {
	provider, username, password, _ := getCredentials()
	acc, err := gophercloud.Authenticate(
		provider,
		gophercloud.AuthOptions{
			Username:    username,
			Password:    password,
			AllowReauth: ar,
		},
	)
	if err != nil {
		panic(err)
	}

	f(acc)
}

func withAPIKeyIdentity(ar bool, f func(gophercloud.AccessProvider)) {
	provider, username, _, apiKey := getCredentials()
	acc, err := gophercloud.Authenticate(
		provider,
		gophercloud.AuthOptions{
			Username:    username,
			ApiKey:      apiKey,
			AllowReauth: ar,
		},
	)
	if err != nil {
		panic(err)
	}

	f(acc)
}

// withServerApi acquires the cloud servers API.
func withServerApi(acc gophercloud.AccessProvider, f func(gophercloud.CloudServersProvider)) {
	api, err := gophercloud.ServersApi(acc, gophercloud.ApiCriteria{
		Name:      "cloudServersOpenStack",
		VersionId: "2",
		UrlChoice: gophercloud.PublicURL,
	})
	if err != nil {
		panic(err)
	}

	f(api)
}

// waitForServerState polls, every 10 seconds, for a given server to appear in the indicated state.
// This call will block forever if it never appears in the desired state, so if a timeout is required,
// make sure to call this function in a goroutine.
func waitForServerState(api gophercloud.CloudServersProvider, id, state string) error {
	for {
		s, err := api.ServerById(id)
		if err != nil {
			return err
		}
		if s.Status == state {
			return nil
		}
		time.Sleep(10 * time.Second)
	}
	panic("Impossible")
}

// waitForImageState polls, every 10 seconds, for a given image to appear in the indicated state.
// This call will block forever if it never appears in the desired state, so if a timeout is required,
// make sure to call this function in a goroutine.
func waitForImageState(api gophercloud.CloudServersProvider, id, state string) error {
	for {
		s, err := api.ImageById(id)
		if err != nil {
			return err
		}
		if s.Status == state {
			return nil
		}
		time.Sleep(10 * time.Second)
	}
	panic("Impossible")
}
