package sign

import (
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"time"
)

func examplePEMReader() io.Reader {
	reader, err := generatePEM(randReader, nil)
	if err != nil {
		panic(fmt.Sprintf("Unexpected pem generation err %v", err))
	}

	return reader
}

func ExampleCookieSigner_Sign() {
	origRandReader := randReader
	randReader = newRandomReader(rand.New(rand.NewSource(1)))
	defer func() {
		randReader = origRandReader
	}()

	// Load your private key so it can be used by the CookieSigner
	// To load private key from file use `sign.LoadPEMPrivKeyFile`.
	privKey, err := LoadPEMPrivKey(examplePEMReader())
	if err != nil {
		fmt.Println("failed to load private key", err)
		return
	}

	cookieSigner := NewCookieSigner("keyID", privKey)

	// Use the signer to sign the URL
	cookies, err := cookieSigner.Sign("http://example.com/somepath/*", testSignTime.Add(30*time.Minute))
	if err != nil {
		fmt.Println("failed to sign cookies with policy,", err)
		return
	}

	printExampleCookies(cookies)
	// Output:
	// Cookies:
	// CloudFront-Policy: eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cDovL2V4YW1wbGUuY29tL3NvbWVwYXRoLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjEyNTc4OTU4MDB9fX1dfQ__, , , false
	// CloudFront-Signature: o~jvj~CFkvGZB~yYED3elicKZag-CRijy8yD2E5yF1s7VNV7kNeQWC7MDtEcBQ8-eh7Xgjh0wMPQdAVdh09gBObd-hXDpKUyh8YKxogj~oloV~8KOvqE5xzWiKcqjdfJjmT5iEqIui~H1ExYjyKjgir79npmlyYkaJS5s62EQa8_, , , false
	// CloudFront-Key-Pair-Id: keyID, , , false
}

func ExampleCookieSigner_SignWithPolicy() {
	origRandReader := randReader
	randReader = newRandomReader(rand.New(rand.NewSource(1)))
	defer func() {
		randReader = origRandReader
	}()

	// Sign cookie to be valid for 30 minutes from now, expires one hour
	// from now, and restricted to the 192.0.2.0/24 IP address range.
	// http://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-setting-signed-cookie-custom-policy.html
	p := &Policy{
		// Only a single policy statement can be used with CloudFront
		// cookie signatures.
		Statements: []Statement{{
			// Read the provided documentation on how to set this correctly,
			// you'll probably want to use wildcards
			Resource: "http://sub.cloudfront.com",
			Condition: Condition{
				// Optional IP source address range
				IPAddress: &IPAddress{SourceIP: "192.0.2.0/24"},
				// Optional date URL is not valid until
				DateGreaterThan: &AWSEpochTime{testSignTime.Add(30 * time.Minute)},
				// Required date the URL will expire after
				DateLessThan: &AWSEpochTime{testSignTime.Add(1 * time.Hour)},
			},
		},
		},
	}

	// Load your private key so it can be used by the CookieSigner
	// To load private key from file use `sign.LoadPEMPrivKeyFile`.
	privKey, err := LoadPEMPrivKey(examplePEMReader())
	if err != nil {
		fmt.Println("failed to load private key", err)
		return
	}

	// Key ID that represents the key pair associated with the private key
	keyID := "privateKeyID"

	// Set credentials to the CookieSigner.
	cookieSigner := NewCookieSigner(keyID, privKey)

	// Avoid adding an Expire or MaxAge. See provided AWS Documentation for
	// more info.
	cookies, err := cookieSigner.SignWithPolicy(p)
	if err != nil {
		fmt.Println("failed to sign cookies with policy,", err)
		return
	}

	printExampleCookies(cookies)
	// Output:
	// Cookies:
	// CloudFront-Policy: eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cDovL3N1Yi5jbG91ZGZyb250LmNvbSIsIkNvbmRpdGlvbiI6eyJJcEFkZHJlc3MiOnsiQVdTOlNvdXJjZUlwIjoiMTkyLjAuMi4wLzI0In0sIkRhdGVHcmVhdGVyVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxMjU3ODk1ODAwfSwiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjEyNTc4OTc2MDB9fX1dfQ__, , , false
	// CloudFront-Signature: JaWdcbr98colrDAhOpkyxqCZev2IAxURu1RKKo1wS~sI5XdNXWYbZJs2FdpbJ475ZvmhZ1-r4ENUqBXAlRfPfOc21Hm4~24jRmPTO3512D4uuJHrPVxSfgeGuFeigfCGWAqyfYYH1DsFl5JQDpzetsNI3ZhGRkQb8V-oYFanddg_, , , false
	// CloudFront-Key-Pair-Id: privateKeyID, , , false
}

func ExampleCookieOptions() {
	origRandReader := randReader
	randReader = newRandomReader(rand.New(rand.NewSource(1)))
	defer func() {
		randReader = origRandReader
	}()

	// Load your private key so it can be used by the CookieSigner
	// To load private key from file use `sign.LoadPEMPrivKeyFile`.
	privKey, err := LoadPEMPrivKey(examplePEMReader())
	if err != nil {
		fmt.Println("failed to load private key", err)
		return
	}

	// Create the CookieSigner with options set. These options can be set
	// directly with cookieSigner.Opts. These values can be overridden on
	// individual Sign and SignWithProfile calls.
	cookieSigner := NewCookieSigner("keyID", privKey, func(o *CookieOptions) {
		//provide an optional struct fields to specify other options
		o.Path = "/"

		// http://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/CNAMEs.html
		o.Domain = ".cNameAssociatedWithMyDistribution.com"

		// Make sure your app/site can handle https payloads, otherwise
		// set this to false.
		o.Secure = true
	})

	// Use the signer to sign the URL
	cookies, err := cookieSigner.Sign("http*://*", testSignTime.Add(30*time.Minute), func(o *CookieOptions) {
		o.Path = "/mypath/"
	})
	if err != nil {
		fmt.Println("failed to sign cookies with policy,", err)
		return
	}

	printExampleCookies(cookies)
	// Output:
	// Cookies:
	// CloudFront-Policy: eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cCo6Ly8qIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxMjU3ODk1ODAwfX19XX0_, /mypath/, .cNameAssociatedWithMyDistribution.com, true
	// CloudFront-Signature: Yco06vgowwvSYgTSY9XbXpBcTlUlqpyyYXgRhus3nfnC74A7oQ~fMBH0we-rGxvph8ZyHnTxC5ubbPKSzo3EHUm2IcQeEo4p6WCgZZMzCuLlkpeMKhMAkCqX7rmUfkXhTslBHe~ylcmaZqo-hdnOiWrXk2U974ZQbbt5cOjwQG0_, /mypath/, .cNameAssociatedWithMyDistribution.com, true
	// CloudFront-Key-Pair-Id: keyID, /mypath/, .cNameAssociatedWithMyDistribution.com, true
}

func printExampleCookies(cookies []*http.Cookie) {
	fmt.Println("Cookies:")
	for _, c := range cookies {
		fmt.Printf("%s: %s, %s, %s, %t\n", c.Name, c.Value, c.Path, c.Domain, c.Secure)
	}
}
