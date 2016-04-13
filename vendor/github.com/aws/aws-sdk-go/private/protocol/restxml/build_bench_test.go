// +build bench

package restxml_test

import (
	"testing"

	"bytes"
	"encoding/xml"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/private/protocol/restxml"
	"github.com/aws/aws-sdk-go/service/cloudfront"
)

func BenchmarkRESTXMLBuild_Complex_cloudfrontCreateDistribution(b *testing.B) {
	params := restxmlBuildCreateDistroParms

	op := &request.Operation{
		Name:       "CreateDistribution",
		HTTPMethod: "POST",
		HTTPPath:   "/2015-04-17/distribution/{DistributionId}/invalidation",
	}

	benchRESTXMLBuild(b, op, params)
}

func BenchmarkRESTXMLBuild_Simple_cloudfrontDeleteStreamingDistribution(b *testing.B) {
	params := &cloudfront.DeleteDistributionInput{
		Id:      aws.String("string"), // Required
		IfMatch: aws.String("string"),
	}
	op := &request.Operation{
		Name:       "DeleteStreamingDistribution",
		HTTPMethod: "DELETE",
		HTTPPath:   "/2015-04-17/streaming-distribution/{Id}",
	}
	benchRESTXMLBuild(b, op, params)
}

func BenchmarkEncodingXMLMarshal_Simple_cloudfrontDeleteStreamingDistribution(b *testing.B) {
	params := &cloudfront.DeleteDistributionInput{
		Id:      aws.String("string"), // Required
		IfMatch: aws.String("string"),
	}

	for i := 0; i < b.N; i++ {
		buf := &bytes.Buffer{}
		encoder := xml.NewEncoder(buf)
		if err := encoder.Encode(params); err != nil {
			b.Fatal("Unexpected error", err)
		}
	}
}

func benchRESTXMLBuild(b *testing.B, op *request.Operation, params interface{}) {
	svc := awstesting.NewClient()
	svc.ServiceName = "cloudfront"
	svc.APIVersion = "2015-04-17"

	for i := 0; i < b.N; i++ {
		r := svc.NewRequest(op, params, nil)
		restxml.Build(r)
		if r.Error != nil {
			b.Fatal("Unexpected error", r.Error)
		}
	}
}

var restxmlBuildCreateDistroParms = &cloudfront.CreateDistributionInput{
	DistributionConfig: &cloudfront.DistributionConfig{ // Required
		CallerReference: aws.String("string"), // Required
		Comment:         aws.String("string"), // Required
		DefaultCacheBehavior: &cloudfront.DefaultCacheBehavior{ // Required
			ForwardedValues: &cloudfront.ForwardedValues{ // Required
				Cookies: &cloudfront.CookiePreference{ // Required
					Forward: aws.String("ItemSelection"), // Required
					WhitelistedNames: &cloudfront.CookieNames{
						Quantity: aws.Int64(1), // Required
						Items: []*string{
							aws.String("string"), // Required
							// More values...
						},
					},
				},
				QueryString: aws.Bool(true), // Required
				Headers: &cloudfront.Headers{
					Quantity: aws.Int64(1), // Required
					Items: []*string{
						aws.String("string"), // Required
						// More values...
					},
				},
			},
			MinTTL:         aws.Int64(1),         // Required
			TargetOriginId: aws.String("string"), // Required
			TrustedSigners: &cloudfront.TrustedSigners{ // Required
				Enabled:  aws.Bool(true), // Required
				Quantity: aws.Int64(1),   // Required
				Items: []*string{
					aws.String("string"), // Required
					// More values...
				},
			},
			ViewerProtocolPolicy: aws.String("ViewerProtocolPolicy"), // Required
			AllowedMethods: &cloudfront.AllowedMethods{
				Items: []*string{ // Required
					aws.String("Method"), // Required
					// More values...
				},
				Quantity: aws.Int64(1), // Required
				CachedMethods: &cloudfront.CachedMethods{
					Items: []*string{ // Required
						aws.String("Method"), // Required
						// More values...
					},
					Quantity: aws.Int64(1), // Required
				},
			},
			DefaultTTL:      aws.Int64(1),
			MaxTTL:          aws.Int64(1),
			SmoothStreaming: aws.Bool(true),
		},
		Enabled: aws.Bool(true), // Required
		Origins: &cloudfront.Origins{ // Required
			Quantity: aws.Int64(1), // Required
			Items: []*cloudfront.Origin{
				{ // Required
					DomainName: aws.String("string"), // Required
					Id:         aws.String("string"), // Required
					CustomOriginConfig: &cloudfront.CustomOriginConfig{
						HTTPPort:             aws.Int64(1),                       // Required
						HTTPSPort:            aws.Int64(1),                       // Required
						OriginProtocolPolicy: aws.String("OriginProtocolPolicy"), // Required
					},
					OriginPath: aws.String("string"),
					S3OriginConfig: &cloudfront.S3OriginConfig{
						OriginAccessIdentity: aws.String("string"), // Required
					},
				},
				// More values...
			},
		},
		Aliases: &cloudfront.Aliases{
			Quantity: aws.Int64(1), // Required
			Items: []*string{
				aws.String("string"), // Required
				// More values...
			},
		},
		CacheBehaviors: &cloudfront.CacheBehaviors{
			Quantity: aws.Int64(1), // Required
			Items: []*cloudfront.CacheBehavior{
				{ // Required
					ForwardedValues: &cloudfront.ForwardedValues{ // Required
						Cookies: &cloudfront.CookiePreference{ // Required
							Forward: aws.String("ItemSelection"), // Required
							WhitelistedNames: &cloudfront.CookieNames{
								Quantity: aws.Int64(1), // Required
								Items: []*string{
									aws.String("string"), // Required
									// More values...
								},
							},
						},
						QueryString: aws.Bool(true), // Required
						Headers: &cloudfront.Headers{
							Quantity: aws.Int64(1), // Required
							Items: []*string{
								aws.String("string"), // Required
								// More values...
							},
						},
					},
					MinTTL:         aws.Int64(1),         // Required
					PathPattern:    aws.String("string"), // Required
					TargetOriginId: aws.String("string"), // Required
					TrustedSigners: &cloudfront.TrustedSigners{ // Required
						Enabled:  aws.Bool(true), // Required
						Quantity: aws.Int64(1),   // Required
						Items: []*string{
							aws.String("string"), // Required
							// More values...
						},
					},
					ViewerProtocolPolicy: aws.String("ViewerProtocolPolicy"), // Required
					AllowedMethods: &cloudfront.AllowedMethods{
						Items: []*string{ // Required
							aws.String("Method"), // Required
							// More values...
						},
						Quantity: aws.Int64(1), // Required
						CachedMethods: &cloudfront.CachedMethods{
							Items: []*string{ // Required
								aws.String("Method"), // Required
								// More values...
							},
							Quantity: aws.Int64(1), // Required
						},
					},
					DefaultTTL:      aws.Int64(1),
					MaxTTL:          aws.Int64(1),
					SmoothStreaming: aws.Bool(true),
				},
				// More values...
			},
		},
		CustomErrorResponses: &cloudfront.CustomErrorResponses{
			Quantity: aws.Int64(1), // Required
			Items: []*cloudfront.CustomErrorResponse{
				{ // Required
					ErrorCode:          aws.Int64(1), // Required
					ErrorCachingMinTTL: aws.Int64(1),
					ResponseCode:       aws.String("string"),
					ResponsePagePath:   aws.String("string"),
				},
				// More values...
			},
		},
		DefaultRootObject: aws.String("string"),
		Logging: &cloudfront.LoggingConfig{
			Bucket:         aws.String("string"), // Required
			Enabled:        aws.Bool(true),       // Required
			IncludeCookies: aws.Bool(true),       // Required
			Prefix:         aws.String("string"), // Required
		},
		PriceClass: aws.String("PriceClass"),
		Restrictions: &cloudfront.Restrictions{
			GeoRestriction: &cloudfront.GeoRestriction{ // Required
				Quantity:        aws.Int64(1),                     // Required
				RestrictionType: aws.String("GeoRestrictionType"), // Required
				Items: []*string{
					aws.String("string"), // Required
					// More values...
				},
			},
		},
		ViewerCertificate: &cloudfront.ViewerCertificate{
			CloudFrontDefaultCertificate: aws.Bool(true),
			IAMCertificateId:             aws.String("string"),
			MinimumProtocolVersion:       aws.String("MinimumProtocolVersion"),
			SSLSupportMethod:             aws.String("SSLSupportMethod"),
		},
	},
}
