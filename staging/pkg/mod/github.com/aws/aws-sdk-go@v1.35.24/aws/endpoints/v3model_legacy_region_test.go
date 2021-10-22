// +build go1.7

package endpoints

import (
	"regexp"
	"testing"
)

func TestEndpointFor_STSRegionalFlag(t *testing.T) {

	// mock STS regional endpoints model
	mockSTSModelPartition := partition{
		ID:        "aws",
		Name:      "AWS Standard",
		DNSSuffix: "amazonaws.com",
		RegionRegex: regionRegex{
			Regexp: func() *regexp.Regexp {
				reg, _ := regexp.Compile("^(us|eu|ap|sa|ca|me)\\-\\w+\\-\\d+$")
				return reg
			}(),
		},
		Defaults: endpoint{
			Hostname:          "{service}.{region}.{dnsSuffix}",
			Protocols:         []string{"https"},
			SignatureVersions: []string{"v4"},
		},
		Regions: regions{
			"ap-east-1": region{
				Description: "Asia Pacific (Hong Kong)",
			},
			"ap-northeast-1": region{
				Description: "Asia Pacific (Tokyo)",
			},
			"ap-northeast-2": region{
				Description: "Asia Pacific (Seoul)",
			},
			"ap-south-1": region{
				Description: "Asia Pacific (Mumbai)",
			},
			"ap-southeast-1": region{
				Description: "Asia Pacific (Singapore)",
			},
			"ap-southeast-2": region{
				Description: "Asia Pacific (Sydney)",
			},
			"ca-central-1": region{
				Description: "Canada (Central)",
			},
			"eu-central-1": region{
				Description: "EU (Frankfurt)",
			},
			"eu-north-1": region{
				Description: "EU (Stockholm)",
			},
			"eu-west-1": region{
				Description: "EU (Ireland)",
			},
			"eu-west-2": region{
				Description: "EU (London)",
			},
			"eu-west-3": region{
				Description: "EU (Paris)",
			},
			"me-south-1": region{
				Description: "Middle East (Bahrain)",
			},
			"sa-east-1": region{
				Description: "South America (Sao Paulo)",
			},
			"us-east-1": region{
				Description: "US East (N. Virginia)",
			},
			"us-east-2": region{
				Description: "US East (Ohio)",
			},
			"us-west-1": region{
				Description: "US West (N. California)",
			},
			"us-west-2": region{
				Description: "US West (Oregon)",
			},
		},
		Services: services{
			"sts": service{
				PartitionEndpoint: "aws-global",
				Defaults:          endpoint{},
				Endpoints: endpoints{
					"ap-east-1":      endpoint{},
					"ap-northeast-1": endpoint{},
					"ap-northeast-2": endpoint{},
					"ap-south-1":     endpoint{},
					"ap-southeast-1": endpoint{},
					"ap-southeast-2": endpoint{},
					"aws-global": endpoint{
						Hostname: "sts.amazonaws.com",
						CredentialScope: credentialScope{
							Region: "us-east-1",
						},
					},
					"ca-central-1": endpoint{},
					"eu-central-1": endpoint{},
					"eu-north-1":   endpoint{},
					"eu-west-1":    endpoint{},
					"eu-west-2":    endpoint{},
					"eu-west-3":    endpoint{},
					"me-south-1":   endpoint{},
					"sa-east-1":    endpoint{},
					"us-east-1":    endpoint{},
					"us-east-1-fips": endpoint{
						Hostname: "sts-fips.us-east-1.amazonaws.com",
						CredentialScope: credentialScope{
							Region: "us-east-1",
						},
					},
					"us-east-2": endpoint{},
					"us-east-2-fips": endpoint{
						Hostname: "sts-fips.us-east-2.amazonaws.com",
						CredentialScope: credentialScope{
							Region: "us-east-2",
						},
					},
					"us-west-1": endpoint{},
					"us-west-1-fips": endpoint{
						Hostname: "sts-fips.us-west-1.amazonaws.com",
						CredentialScope: credentialScope{
							Region: "us-west-1",
						},
					},
					"us-west-2": endpoint{},
					"us-west-2-fips": endpoint{
						Hostname: "sts-fips.us-west-2.amazonaws.com",
						CredentialScope: credentialScope{
							Region: "us-west-2",
						},
					},
				},
			},
		},
	}

	// resolver for mock STS regional endpoints model
	resolver := mockSTSModelPartition

	cases := map[string]struct {
		service, region                                     string
		regional                                            bool
		ExpectURL, ExpectSigningMethod, ExpectSigningRegion string
		ExpectSigningNameDerived                            bool
	}{
		// STS Endpoints resolver tests :
		"sts/us-west-2/regional": {
			service:                  "sts",
			region:                   "us-west-2",
			regional:                 true,
			ExpectURL:                "https://sts.us-west-2.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-west-2",
		},
		"sts/us-west-2/legacy": {
			service:                  "sts",
			region:                   "us-west-2",
			regional:                 false,
			ExpectURL:                "https://sts.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-east-1",
		},
		"sts/ap-east-1/regional": {
			service:                  "sts",
			region:                   "ap-east-1",
			regional:                 true,
			ExpectURL:                "https://sts.ap-east-1.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "ap-east-1",
		},
		"sts/ap-east-1/legacy": {
			service:                  "sts",
			region:                   "ap-east-1",
			regional:                 false,
			ExpectURL:                "https://sts.ap-east-1.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "ap-east-1",
		},
		"sts/us-west-2-fips/regional": {
			service:                  "sts",
			region:                   "us-west-2-fips",
			regional:                 true,
			ExpectURL:                "https://sts-fips.us-west-2.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-west-2",
		},
		"sts/us-west-2-fips/legacy": {
			service:                  "sts",
			region:                   "us-west-2-fips",
			regional:                 false,
			ExpectURL:                "https://sts-fips.us-west-2.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-west-2",
		},
		"sts/aws-global/regional": {
			service:                  "sts",
			region:                   "aws-global",
			regional:                 true,
			ExpectURL:                "https://sts.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-east-1",
		},
		"sts/aws-global/legacy": {
			service:                  "sts",
			region:                   "aws-global",
			regional:                 false,
			ExpectURL:                "https://sts.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-east-1",
		},
		"sts/ap-south-1/regional": {
			service:                  "sts",
			region:                   "ap-south-1",
			regional:                 true,
			ExpectURL:                "https://sts.ap-south-1.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "ap-south-1",
		},
		"sts/ap-south-1/legacy": {
			service:                  "sts",
			region:                   "ap-south-1",
			regional:                 false,
			ExpectURL:                "https://sts.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-east-1",
		},
		"sts/ap-northeast-1/regional": {
			service:                  "sts",
			region:                   "ap-northeast-1",
			regional:                 true,
			ExpectURL:                "https://sts.ap-northeast-1.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "ap-northeast-1",
		},
		"sts/ap-northeast-1/legacy": {
			service:                  "sts",
			region:                   "ap-northeast-1",
			regional:                 false,
			ExpectURL:                "https://sts.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-east-1",
		},
		"sts/ap-southeast-1/regional": {
			service:                  "sts",
			region:                   "ap-southeast-1",
			regional:                 true,
			ExpectURL:                "https://sts.ap-southeast-1.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "ap-southeast-1",
		},
		"sts/ap-southeast-1/legacy": {
			service:                  "sts",
			region:                   "ap-southeast-1",
			regional:                 false,
			ExpectURL:                "https://sts.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-east-1",
		},
		"sts/ca-central-1/regional": {
			service:                  "sts",
			region:                   "ca-central-1",
			regional:                 true,
			ExpectURL:                "https://sts.ca-central-1.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "ca-central-1",
		},
		"sts/ca-central-1/legacy": {
			service:                  "sts",
			region:                   "ca-central-1",
			regional:                 false,
			ExpectURL:                "https://sts.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-east-1",
		},
		"sts/eu-central-1/regional": {
			service:                  "sts",
			region:                   "eu-central-1",
			regional:                 true,
			ExpectURL:                "https://sts.eu-central-1.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "eu-central-1",
		},
		"sts/eu-central-1/legacy": {
			service:                  "sts",
			region:                   "eu-central-1",
			regional:                 false,
			ExpectURL:                "https://sts.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-east-1",
		},
		"sts/eu-north-1/regional": {
			service:                  "sts",
			region:                   "eu-north-1",
			regional:                 true,
			ExpectURL:                "https://sts.eu-north-1.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "eu-north-1",
		},
		"sts/eu-north-1/legacy": {
			service:                  "sts",
			region:                   "eu-north-1",
			regional:                 false,
			ExpectURL:                "https://sts.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-east-1",
		},
		"sts/eu-west-1/regional": {
			service:                  "sts",
			region:                   "eu-west-1",
			regional:                 true,
			ExpectURL:                "https://sts.eu-west-1.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "eu-west-1",
		},
		"sts/eu-west-1/legacy": {
			service:                  "sts",
			region:                   "eu-west-1",
			regional:                 false,
			ExpectURL:                "https://sts.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-east-1",
		},
		"sts/eu-west-2/regional": {
			service:                  "sts",
			region:                   "eu-west-2",
			regional:                 true,
			ExpectURL:                "https://sts.eu-west-2.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "eu-west-2",
		},
		"sts/eu-west-2/legacy": {
			service:                  "sts",
			region:                   "eu-west-2",
			regional:                 false,
			ExpectURL:                "https://sts.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-east-1",
		},
		"sts/eu-west-3/regional": {
			service:                  "sts",
			region:                   "eu-west-3",
			regional:                 true,
			ExpectURL:                "https://sts.eu-west-3.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "eu-west-3",
		},
		"sts/eu-west-3/legacy": {
			service:                  "sts",
			region:                   "eu-west-3",
			regional:                 false,
			ExpectURL:                "https://sts.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-east-1",
		},
		"sts/sa-east-1/regional": {
			service:                  "sts",
			region:                   "sa-east-1",
			regional:                 true,
			ExpectURL:                "https://sts.sa-east-1.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "sa-east-1",
		},
		"sts/sa-east-1/legacy": {
			service:                  "sts",
			region:                   "sa-east-1",
			regional:                 false,
			ExpectURL:                "https://sts.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-east-1",
		},
		"sts/us-east-1/regional": {
			service:                  "sts",
			region:                   "us-east-1",
			regional:                 true,
			ExpectURL:                "https://sts.us-east-1.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-east-1",
		},
		"sts/us-east-1/legacy": {
			service:                  "sts",
			region:                   "us-east-1",
			regional:                 false,
			ExpectURL:                "https://sts.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-east-1",
		},
		"sts/us-east-2/regional": {
			service:                  "sts",
			region:                   "us-east-2",
			regional:                 true,
			ExpectURL:                "https://sts.us-east-2.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-east-2",
		},
		"sts/us-east-2/legacy": {
			service:                  "sts",
			region:                   "us-east-2",
			regional:                 false,
			ExpectURL:                "https://sts.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-east-1",
		},
		"sts/us-west-1/regional": {
			service:                  "sts",
			region:                   "us-west-1",
			regional:                 true,
			ExpectURL:                "https://sts.us-west-1.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-west-1",
		},
		"sts/us-west-1/legacy": {
			service:                  "sts",
			region:                   "us-west-1",
			regional:                 false,
			ExpectURL:                "https://sts.amazonaws.com",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "us-east-1",
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			var optionSlice []func(o *Options)
			optionSlice = append(optionSlice, func(o *Options) {
				if c.regional {
					o.STSRegionalEndpoint = RegionalSTSEndpoint
				}
			})

			actual, err := resolver.EndpointFor(c.service, c.region, optionSlice...)
			if err != nil {
				t.Fatalf("failed to resolve endpoint, %v", err)
			}

			if e, a := c.ExpectURL, actual.URL; e != a {
				t.Errorf("expect %v, got %v", e, a)
			}

			if e, a := c.ExpectSigningMethod, actual.SigningMethod; e != a {
				t.Errorf("expect %v, got %v", e, a)
			}

			if e, a := c.ExpectSigningNameDerived, actual.SigningNameDerived; e != a {
				t.Errorf("expect %v, got %v", e, a)
			}

			if e, a := c.ExpectSigningRegion, actual.SigningRegion; e != a {
				t.Errorf("expect %v, got %v", e, a)
			}

		})
	}
}

func TestEndpointFor_S3UsEast1RegionalFlag(t *testing.T) {

	// mock S3 regional endpoints model
	mockS3ModelPartition := partition{
		ID:        "aws",
		Name:      "AWS Standard",
		DNSSuffix: "amazonaws.com",
		RegionRegex: regionRegex{
			Regexp: func() *regexp.Regexp {
				reg, _ := regexp.Compile("^(us|eu|ap|sa|ca|me)\\-\\w+\\-\\d+$")
				return reg
			}(),
		},
		Defaults: endpoint{
			Hostname:          "{service}.{region}.{dnsSuffix}",
			Protocols:         []string{"https"},
			SignatureVersions: []string{"v4"},
		},
		Regions: regions{
			"ap-east-1": region{
				Description: "Asia Pacific (Hong Kong)",
			},
			"ap-northeast-1": region{
				Description: "Asia Pacific (Tokyo)",
			},
			"ap-northeast-2": region{
				Description: "Asia Pacific (Seoul)",
			},
			"ap-south-1": region{
				Description: "Asia Pacific (Mumbai)",
			},
			"ap-southeast-1": region{
				Description: "Asia Pacific (Singapore)",
			},
			"ap-southeast-2": region{
				Description: "Asia Pacific (Sydney)",
			},
			"ca-central-1": region{
				Description: "Canada (Central)",
			},
			"eu-central-1": region{
				Description: "EU (Frankfurt)",
			},
			"eu-north-1": region{
				Description: "EU (Stockholm)",
			},
			"eu-west-1": region{
				Description: "EU (Ireland)",
			},
			"eu-west-2": region{
				Description: "EU (London)",
			},
			"eu-west-3": region{
				Description: "EU (Paris)",
			},
			"me-south-1": region{
				Description: "Middle East (Bahrain)",
			},
			"sa-east-1": region{
				Description: "South America (Sao Paulo)",
			},
			"us-east-1": region{
				Description: "US East (N. Virginia)",
			},
			"us-east-2": region{
				Description: "US East (Ohio)",
			},
			"us-west-1": region{
				Description: "US West (N. California)",
			},
			"us-west-2": region{
				Description: "US West (Oregon)",
			},
		},
		Services: services{
			"s3": service{
				PartitionEndpoint: "aws-global",
				IsRegionalized:    boxedTrue,
				Defaults: endpoint{
					Protocols:         []string{"http", "https"},
					SignatureVersions: []string{"s3v4"},

					HasDualStack:      boxedTrue,
					DualStackHostname: "{service}.dualstack.{region}.{dnsSuffix}",
				},
				Endpoints: endpoints{
					"ap-east-1": endpoint{},
					"ap-northeast-1": endpoint{
						Hostname:          "s3.ap-northeast-1.amazonaws.com",
						SignatureVersions: []string{"s3", "s3v4"},
					},
					"ap-northeast-2": endpoint{},
					"ap-northeast-3": endpoint{},
					"ap-south-1":     endpoint{},
					"ap-southeast-1": endpoint{
						Hostname:          "s3.ap-southeast-1.amazonaws.com",
						SignatureVersions: []string{"s3", "s3v4"},
					},
					"ap-southeast-2": endpoint{
						Hostname:          "s3.ap-southeast-2.amazonaws.com",
						SignatureVersions: []string{"s3", "s3v4"},
					},
					"aws-global": endpoint{
						Hostname: "s3.amazonaws.com",
						CredentialScope: credentialScope{
							Region: "us-east-1",
						},
					},
					"ca-central-1": endpoint{},
					"eu-central-1": endpoint{},
					"eu-north-1":   endpoint{},
					"eu-west-1": endpoint{
						Hostname:          "s3.eu-west-1.amazonaws.com",
						SignatureVersions: []string{"s3", "s3v4"},
					},
					"eu-west-2":  endpoint{},
					"eu-west-3":  endpoint{},
					"me-south-1": endpoint{},
					"s3-external-1": endpoint{
						Hostname:          "s3-external-1.amazonaws.com",
						SignatureVersions: []string{"s3", "s3v4"},
						CredentialScope: credentialScope{
							Region: "us-east-1",
						},
					},
					"sa-east-1": endpoint{
						Hostname:          "s3.sa-east-1.amazonaws.com",
						SignatureVersions: []string{"s3", "s3v4"},
					},
					"us-east-1": endpoint{
						Hostname:          "s3.us-east-1.amazonaws.com",
						SignatureVersions: []string{"s3", "s3v4"},
					},
					"us-east-2": endpoint{},
					"us-west-1": endpoint{
						Hostname:          "s3.us-west-1.amazonaws.com",
						SignatureVersions: []string{"s3", "s3v4"},
					},
					"us-west-2": endpoint{
						Hostname:          "s3.us-west-2.amazonaws.com",
						SignatureVersions: []string{"s3", "s3v4"},
					},
				},
			},
		},
	}

	// resolver for mock S3 regional endpoints model
	resolver := mockS3ModelPartition

	cases := map[string]struct {
		service, region     string
		regional            S3UsEast1RegionalEndpoint
		ExpectURL           string
		ExpectSigningRegion string
	}{
		// S3 Endpoints resolver tests:
		"s3/us-east-1/regional": {
			service:             "s3",
			region:              "us-east-1",
			regional:            RegionalS3UsEast1Endpoint,
			ExpectURL:           "https://s3.us-east-1.amazonaws.com",
			ExpectSigningRegion: "us-east-1",
		},
		"s3/us-east-1/legacy": {
			service:             "s3",
			region:              "us-east-1",
			ExpectURL:           "https://s3.amazonaws.com",
			ExpectSigningRegion: "us-east-1",
		},
		"s3/us-west-1/regional": {
			service:             "s3",
			region:              "us-west-1",
			regional:            RegionalS3UsEast1Endpoint,
			ExpectURL:           "https://s3.us-west-1.amazonaws.com",
			ExpectSigningRegion: "us-west-1",
		},
		"s3/us-west-1/legacy": {
			service:             "s3",
			region:              "us-west-1",
			regional:            RegionalS3UsEast1Endpoint,
			ExpectURL:           "https://s3.us-west-1.amazonaws.com",
			ExpectSigningRegion: "us-west-1",
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			var optionSlice []func(o *Options)
			optionSlice = append(optionSlice, func(o *Options) {
				o.S3UsEast1RegionalEndpoint = c.regional
			})

			actual, err := resolver.EndpointFor(c.service, c.region, optionSlice...)
			if err != nil {
				t.Fatalf("failed to resolve endpoint, %v", err)
			}

			if e, a := c.ExpectURL, actual.URL; e != a {
				t.Errorf("expect %v, got %v", e, a)
			}

			if e, a := c.ExpectSigningRegion, actual.SigningRegion; e != a {
				t.Errorf("expect %v, got %v", e, a)
			}

		})
	}
}

func TestSTSRegionalEndpoint_CNPartition(t *testing.T) {
	mockSTSCNPartition := partition{
		ID:        "aws-cn",
		Name:      "AWS China",
		DNSSuffix: "amazonaws.com.cn",
		RegionRegex: regionRegex{
			Regexp: func() *regexp.Regexp {
				reg, _ := regexp.Compile("^cn\\-\\w+\\-\\d+$")
				return reg
			}(),
		},
		Defaults: endpoint{
			Hostname:          "{service}.{region}.{dnsSuffix}",
			Protocols:         []string{"https"},
			SignatureVersions: []string{"v4"},
		},
		Regions: regions{
			"cn-north-1": region{
				Description: "China (Beijing)",
			},
			"cn-northwest-1": region{
				Description: "China (Ningxia)",
			},
		},
		Services: services{
			"sts": service{
				Endpoints: endpoints{
					"cn-north-1":     endpoint{},
					"cn-northwest-1": endpoint{},
				},
			},
		},
	}

	resolver := mockSTSCNPartition

	cases := map[string]struct {
		service, region                                     string
		regional                                            bool
		ExpectURL, ExpectSigningMethod, ExpectSigningRegion string
		ExpectSigningNameDerived                            bool
	}{
		"sts/cn-north-1/regional": {
			service:                  "sts",
			region:                   "cn-north-1",
			regional:                 true,
			ExpectURL:                "https://sts.cn-north-1.amazonaws.com.cn",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "cn-north-1",
		},
		"sts/cn-north-1/legacy": {
			service:                  "sts",
			region:                   "cn-north-1",
			regional:                 false,
			ExpectURL:                "https://sts.cn-north-1.amazonaws.com.cn",
			ExpectSigningMethod:      "v4",
			ExpectSigningNameDerived: true,
			ExpectSigningRegion:      "cn-north-1",
		},
	}

	for name, c := range cases {
		var optionSlice []func(o *Options)
		t.Run(name, func(t *testing.T) {
			if c.regional {
				optionSlice = append(optionSlice, STSRegionalEndpointOption)
			}
			actual, err := resolver.EndpointFor(c.service, c.region, optionSlice...)
			if err != nil {
				t.Fatalf("failed to resolve endpoint, %v", err)
			}

			if e, a := c.ExpectURL, actual.URL; e != a {
				t.Errorf("expect %v, got %v", e, a)
			}
			if e, a := c.ExpectSigningMethod, actual.SigningMethod; e != a {
				t.Errorf("expect %v, got %v", e, a)
			}
			if e, a := c.ExpectSigningNameDerived, actual.SigningNameDerived; e != a {
				t.Errorf("expect %v, got %v", e, a)
			}
			if e, a := c.ExpectSigningRegion, actual.SigningRegion; e != a {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}

}
