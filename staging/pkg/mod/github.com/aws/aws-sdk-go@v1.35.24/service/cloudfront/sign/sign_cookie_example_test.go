package sign

import (
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go/awstesting/mock"
)

func examplePEMReader() io.Reader {
	reader, err := generatePEM(randReader, nil)
	if err != nil {
		panic(fmt.Sprintf("Unexpected pem generation err %v", err))
	}

	return reader
}

func ExampleCookieSigner_Sign() {
	// Load your private key so it can be used by the CookieSigner
	// To load private key from file use `sign.LoadPEMPrivKeyFile`.
	privKey := mock.RSAPrivateKey

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
	// CloudFront-Signature: Gx67J8t1VanOFWN84BQlpN064aGCicJv916esnPr9Rdb2RKEzl7VoDOsh9Uez7SY5blWATkN5F3xNicTpOupdN-ywrTf5zCTLz5RmvLrIyEDS3Y1knTGoWvp6nnIb9FOuI1rSyBaJ8VKuNVQGmvqzXGXsnipgSBPjpkL6Ja3dBXeKIbUeaLKQBZrtMWv9nS5VyG4nOP-CRcTgQ5DA3-h~WP2ZzhONb6yoYXeOSvBu8HBl0IZI27InLpxiKlkWUchNncnkZ32Md0CwLLrA4wxFl0fYsxxg6Us2XBYRGmudugJHgkkopem9Cc4eOiDGMABcJGAuZprVXT0WuOBYJngTA__, , , false
	// CloudFront-Key-Pair-Id: keyID, , , false
}

func ExampleCookieSigner_SignWithPolicy() {
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
	privKey := mock.RSAPrivateKey

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
	// CloudFront-Signature: Ixn4bF1LLrLcB8XG-t5bZbIB0vfwSF2s4gkef~PcNBdx73MVvZD3v8DZ5GzcqNrybMiqdYJY5KqK6vTsf5JXDgwFFz-h98wdsbV-izcuonPdzMHp4Ay4qyXM6Ed5jB9dUWYGwMkA6rsWXpftfX8xmk4tG1LwFuJV6nAsx4cfpuKwo4vU2Hyr2-fkA7MZG8AHkpDdVUnjm1q-Re9HdG0nCq-2lnBAdOchBpJt37narOj-Zg6cbx~6rzQLVQd8XIv-Bn7VTc1tkBAJVtGOHb0q~PLzSRmtNGYTnpL0z~gp3tq8lhZc2HuvJW5-tZaYP9yufeIzk5bqsT6DT4iDuclKKw__, , , false
	// CloudFront-Key-Pair-Id: privateKeyID, , , false
}

func ExampleCookieOptions() {
	privKey := mock.RSAPrivateKey

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
	// CloudFront-Signature: DBXEcU6NoyAelecgEcr6mE1IHCqqlHdGwAC2X1dYn0QOLZ8Ar~oehlMub~hEh~UEMijR15ii-yUYf-3ML0b1SwWkh4rTa-SFURWDVuu~vW3cQzRZ4wQrgDR3DGJINrtGtEsDSzA6zdwtZsfvc1W9IRPn9rnVmwDdUurSrcp9M7CdcjkEw9Au~gULX7aUuW87DI5GI7jLo6emmBB1p4V~xAv8rDqOyxdhBzWKDTvl6ErIXnzHitgMclNZrkn-m27BhTQsJOs2R~gT2VrQw-IWX6NMD8r0TDH4DE2HQ8N7jZ0nf8gezbyFk-OhD1P9FUNb1PlwcZWfXtfgHQmM-BmrSQ__, /mypath/, .cNameAssociatedWithMyDistribution.com, true
	// CloudFront-Key-Pair-Id: keyID, /mypath/, .cNameAssociatedWithMyDistribution.com, true
}

func printExampleCookies(cookies []*http.Cookie) {
	fmt.Println("Cookies:")
	for _, c := range cookies {
		fmt.Printf("%s: %s, %s, %s, %t\n", c.Name, c.Value, c.Path, c.Domain, c.Secure)
	}
}
