// +build codegen

package api

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

type service struct {
	srcName string
	dstName string

	serviceVersion string
}

var mergeServices = map[string]service{
	"dynamodbstreams": {
		dstName: "dynamodb",
		srcName: "streams.dynamodb",
	},
	"wafregional": {
		dstName:        "waf",
		srcName:        "waf-regional",
		serviceVersion: "2015-08-24",
	},
}

var serviceAliaseNames = map[string]string{
	"costandusagereportservice": "CostandUsageReportService",
	"elasticloadbalancing":      "ELB",
	"elasticloadbalancingv2":    "ELBV2",
	"config":                    "ConfigService",
}

func (a *API) setServiceAliaseName() {
	if newName, ok := serviceAliaseNames[a.PackageName()]; ok {
		a.name = newName
	}
}

// customizationPasses Executes customization logic for the API by package name.
func (a *API) customizationPasses() error {
	var svcCustomizations = map[string]func(*API) error{
		"s3":         s3Customizations,
		"s3control":  s3ControlCustomizations,
		"cloudfront": cloudfrontCustomizations,
		"rds":        rdsCustomizations,

		// Disable endpoint resolving for services that require customer
		// to provide endpoint them selves.
		"cloudsearchdomain": disableEndpointResolving,
		"iotdataplane":      disableEndpointResolving,

		// MTurk smoke test is invalid. The service requires AWS account to be
		// linked to Amazon Mechanical Turk Account.
		"mturk": supressSmokeTest,

		// Backfill the authentication type for cognito identity and sts.
		// Removes the need for the customizations in these services.
		"cognitoidentity": backfillAuthType(NoneAuthType,
			"GetId",
			"GetOpenIdToken",
			"UnlinkIdentity",
			"GetCredentialsForIdentity",
		),
		"sts": backfillAuthType(NoneAuthType,
			"AssumeRoleWithSAML",
			"AssumeRoleWithWebIdentity",
		),
	}

	for k := range mergeServices {
		svcCustomizations[k] = mergeServicesCustomizations
	}

	if fn := svcCustomizations[a.PackageName()]; fn != nil {
		err := fn(a)
		if err != nil {
			return fmt.Errorf("service customization pass failure for %s: %v", a.PackageName(), err)
		}
	}

	return nil
}

func supressSmokeTest(a *API) error {
	a.SmokeTests.TestCases = []SmokeTestCase{}
	return nil
}

// Customizes the API generation to replace values specific to S3.
func s3Customizations(a *API) error {
	var strExpires *Shape

	var keepContentMD5Ref = map[string]struct{}{
		"PutObjectInput":  {},
		"UploadPartInput": {},
	}

	for name, s := range a.Shapes {
		// Remove ContentMD5 members unless specified otherwise.
		if _, keep := keepContentMD5Ref[name]; !keep {
			if _, have := s.MemberRefs["ContentMD5"]; have {
				delete(s.MemberRefs, "ContentMD5")
			}
		}

		// Generate getter methods for API operation fields used by customizations.
		for _, refName := range []string{"Bucket", "SSECustomerKey", "CopySourceSSECustomerKey"} {
			if ref, ok := s.MemberRefs[refName]; ok {
				ref.GenerateGetter = true
			}
		}

		// Generate a endpointARN method for the BucketName shape if this is used as an operation input
		if s.UsedAsInput {
			if s.ShapeName == "CreateBucketInput" {
				// For all operations but CreateBucket the BucketName shape
				// needs to be decorated.
				continue
			}
			var endpointARNShape *ShapeRef
			for _, ref := range s.MemberRefs {
				if ref.OrigShapeName != "BucketName" || ref.Shape.Type != "string" {
					continue
				}
				if endpointARNShape != nil {
					return fmt.Errorf("more then one BucketName shape present on shape")
				}
				ref.EndpointARN = true
				endpointARNShape = ref
			}
			if endpointARNShape != nil {
				s.HasEndpointARNMember = true
				a.HasEndpointARN = true
			}
		}

		// Decorate member references that are modeled with the wrong type.
		// Specifically the case where a member was modeled as a string, but is
		// expected to sent across the wire as a base64 value.
		//
		// e.g. S3's SSECustomerKey and CopySourceSSECustomerKey
		for _, refName := range []string{
			"SSECustomerKey",
			"CopySourceSSECustomerKey",
		} {
			if ref, ok := s.MemberRefs[refName]; ok {
				ref.CustomTags = append(ref.CustomTags, ShapeTag{
					"marshal-as", "blob",
				})
			}
		}

		// Expires should be a string not time.Time since the format is not
		// enforced by S3, and any value can be set to this field outside of the SDK.
		if strings.HasSuffix(name, "Output") {
			if ref, ok := s.MemberRefs["Expires"]; ok {
				if strExpires == nil {
					newShape := *ref.Shape
					strExpires = &newShape
					strExpires.Type = "string"
					strExpires.refs = []*ShapeRef{}
				}
				ref.Shape.removeRef(ref)
				ref.Shape = strExpires
				ref.Shape.refs = append(ref.Shape.refs, &s.MemberRef)
			}
		}
	}
	s3CustRemoveHeadObjectModeledErrors(a)

	return nil
}

// S3 HeadObject API call incorrect models NoSuchKey as valid
// error code that can be returned. This operation does not
// return error codes, all error codes are derived from HTTP
// status codes.
//
// aws/aws-sdk-go#1208
func s3CustRemoveHeadObjectModeledErrors(a *API) {
	op, ok := a.Operations["HeadObject"]
	if !ok {
		return
	}
	op.Documentation += `
//
// See http://docs.aws.amazon.com/AmazonS3/latest/API/ErrorResponses.html#RESTErrorResponses
// for more information on returned errors.`
	op.ErrorRefs = []ShapeRef{}
}

// S3 service operations with an AccountId need accessors to be generated for
// them so the fields can be dynamically accessed without reflection.
func s3ControlCustomizations(a *API) error {
	for _, s := range a.Shapes {
		// Generate a endpointARN method for the BucketName shape if this is used as an operation input
		if s.UsedAsInput {
			if s.ShapeName == "CreateBucketInput" || s.ShapeName == "ListRegionalBucketsInput" {
				// For operations CreateBucketInput and ListRegionalBuckets the OutpostID shape
				// needs to be decorated
				var outpostIDMemberShape *ShapeRef
				for memberName, ref := range s.MemberRefs {
					if memberName != "OutpostId" || ref.Shape.Type != "string" {
						continue
					}
					if outpostIDMemberShape != nil {
						return fmt.Errorf("more then one OutpostID shape present on shape")
					}
					ref.OutpostIDMember = true
					outpostIDMemberShape = ref
				}
				if outpostIDMemberShape != nil {
					s.HasOutpostIDMember = true
					a.HasOutpostID = true
				}
				continue
			}

			// List of input shapes that use accesspoint names as arnable fields
			accessPointNameArnables := map[string]struct{}{
				"GetAccessPointInput":          {},
				"DeleteAccessPointInput":       {},
				"PutAccessPointPolicyInput":    {},
				"GetAccessPointPolicyInput":    {},
				"DeleteAccessPointPolicyInput": {},
			}

			var endpointARNShape *ShapeRef
			for _, ref := range s.MemberRefs {
				// Operations that have AccessPointName field that takes in an ARN as input
				if _, ok := accessPointNameArnables[s.ShapeName]; ok {
					if ref.OrigShapeName != "AccessPointName" || ref.Shape.Type != "string" {
						continue
					}
				} else if ref.OrigShapeName != "BucketName" || ref.Shape.Type != "string" {
					// All other operations currently allow BucketName field to take in ARN.
					// Exceptions for these are CreateBucket and ListRegionalBucket which use
					// Outpost id and are handled above separately.
					continue
				}

				if endpointARNShape != nil {
					return fmt.Errorf("more then one member present on shape takes arn as input")
				}
				ref.EndpointARN = true
				endpointARNShape = ref
			}
			if endpointARNShape != nil {
				s.HasEndpointARNMember = true
				a.HasEndpointARN = true

				for _, ref := range s.MemberRefs {
					// check for account id customization
					if ref.OrigShapeName == "AccountId" && ref.Shape.Type == "string" {
						ref.AccountIDMemberWithARN = true
						s.HasAccountIdMemberWithARN = true
						a.HasAccountIdWithARN = true
					}
				}
			}
		}
	}

	return nil
}

// cloudfrontCustomizations customized the API generation to replace values
// specific to CloudFront.
func cloudfrontCustomizations(a *API) error {
	// MaxItems members should always be integers
	for _, s := range a.Shapes {
		if ref, ok := s.MemberRefs["MaxItems"]; ok {
			ref.ShapeName = "Integer"
			ref.Shape = a.Shapes["Integer"]
		}
	}
	return nil
}

// mergeServicesCustomizations references any duplicate shapes from DynamoDB
func mergeServicesCustomizations(a *API) error {
	info := mergeServices[a.PackageName()]

	p := strings.Replace(a.path, info.srcName, info.dstName, -1)

	if info.serviceVersion != "" {
		index := strings.LastIndex(p, string(filepath.Separator))
		files, _ := ioutil.ReadDir(p[:index])
		if len(files) > 1 {
			panic("New version was introduced")
		}
		p = p[:index] + "/" + info.serviceVersion
	}

	file := filepath.Join(p, "api-2.json")

	serviceAPI := API{}
	serviceAPI.Attach(file)
	serviceAPI.Setup()

	for n := range a.Shapes {
		if _, ok := serviceAPI.Shapes[n]; ok {
			a.Shapes[n].resolvePkg = SDKImportRoot + "/service/" + info.dstName
		}
	}

	return nil
}

// rdsCustomizations are customization for the service/rds. This adds non-modeled fields used for presigning.
func rdsCustomizations(a *API) error {
	inputs := []string{
		"CopyDBSnapshotInput",
		"CreateDBInstanceReadReplicaInput",
		"CopyDBClusterSnapshotInput",
		"CreateDBClusterInput",
	}
	for _, input := range inputs {
		if ref, ok := a.Shapes[input]; ok {
			ref.MemberRefs["SourceRegion"] = &ShapeRef{
				Documentation: docstring(`SourceRegion is the source region where the resource exists. This is not sent over the wire and is only used for presigning. This value should always have the same region as the source ARN.`),
				ShapeName:     "String",
				Shape:         a.Shapes["String"],
				Ignore:        true,
			}
			ref.MemberRefs["DestinationRegion"] = &ShapeRef{
				Documentation: docstring(`DestinationRegion is used for presigning the request to a given region.`),
				ShapeName:     "String",
				Shape:         a.Shapes["String"],
			}
		}
	}

	return nil
}

func disableEndpointResolving(a *API) error {
	a.Metadata.NoResolveEndpoint = true
	return nil
}

func backfillAuthType(typ AuthType, opNames ...string) func(*API) error {
	return func(a *API) error {
		for _, opName := range opNames {
			op, ok := a.Operations[opName]
			if !ok {
				panic("unable to backfill auth-type for unknown operation " + opName)
			}
			if v := op.AuthType; len(v) != 0 {
				fmt.Fprintf(os.Stderr, "unable to backfill auth-type for %s, already set, %s", opName, v)
				continue
			}

			op.AuthType = typ
		}

		return nil
	}
}
