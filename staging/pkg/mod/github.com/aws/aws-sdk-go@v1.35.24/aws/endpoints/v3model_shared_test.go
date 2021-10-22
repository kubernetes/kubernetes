package endpoints

import "regexp"

var testPartitions = partitions{
	partition{
		ID:        "part-id",
		Name:      "partitionName",
		DNSSuffix: "amazonaws.com",
		RegionRegex: regionRegex{
			Regexp: func() *regexp.Regexp {
				reg, _ := regexp.Compile("^(us|eu|ap|sa|ca)\\-\\w+\\-\\d+$")
				return reg
			}(),
		},
		Defaults: endpoint{
			Hostname:          "{service}.{region}.{dnsSuffix}",
			Protocols:         []string{"https"},
			SignatureVersions: []string{"v4"},
		},
		Regions: regions{
			"us-east-1": region{
				Description: "region description",
			},
			"us-west-2": region{},
		},
		Services: services{
			"s3": service{},
			"service1": service{
				Defaults: endpoint{
					CredentialScope: credentialScope{
						Service: "service1",
					},
				},
				Endpoints: endpoints{
					"us-east-1": {},
					"us-west-2": {
						HasDualStack:      boxedTrue,
						DualStackHostname: "{service}.dualstack.{region}.{dnsSuffix}",
					},
				},
			},
			"service2": service{
				Defaults: endpoint{
					CredentialScope: credentialScope{
						Service: "service2",
					},
				},
			},
			"httpService": service{
				Defaults: endpoint{
					Protocols: []string{"http"},
				},
			},
			"globalService": service{
				IsRegionalized:    boxedFalse,
				PartitionEndpoint: "aws-global",
				Endpoints: endpoints{
					"aws-global": endpoint{
						CredentialScope: credentialScope{
							Region: "us-east-1",
						},
						Hostname: "globalService.amazonaws.com",
					},
				},
			},
		},
	},
}
