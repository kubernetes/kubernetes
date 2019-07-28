package endpoints

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/aws/aws-sdk-go/aws/awserr"
)

type modelDefinition map[string]json.RawMessage

// A DecodeModelOptions are the options for how the endpoints model definition
// are decoded.
type DecodeModelOptions struct {
	SkipCustomizations bool
}

// Set combines all of the option functions together.
func (d *DecodeModelOptions) Set(optFns ...func(*DecodeModelOptions)) {
	for _, fn := range optFns {
		fn(d)
	}
}

// DecodeModel unmarshals a Regions and Endpoint model definition file into
// a endpoint Resolver. If the file format is not supported, or an error occurs
// when unmarshaling the model an error will be returned.
//
// Casting the return value of this func to a EnumPartitions will
// allow you to get a list of the partitions in the order the endpoints
// will be resolved in.
//
//    resolver, err := endpoints.DecodeModel(reader)
//
//    partitions := resolver.(endpoints.EnumPartitions).Partitions()
//    for _, p := range partitions {
//        // ... inspect partitions
//    }
func DecodeModel(r io.Reader, optFns ...func(*DecodeModelOptions)) (Resolver, error) {
	var opts DecodeModelOptions
	opts.Set(optFns...)

	// Get the version of the partition file to determine what
	// unmarshaling model to use.
	modelDef := modelDefinition{}
	if err := json.NewDecoder(r).Decode(&modelDef); err != nil {
		return nil, newDecodeModelError("failed to decode endpoints model", err)
	}

	var version string
	if b, ok := modelDef["version"]; ok {
		version = string(b)
	} else {
		return nil, newDecodeModelError("endpoints version not found in model", nil)
	}

	if version == "3" {
		return decodeV3Endpoints(modelDef, opts)
	}

	return nil, newDecodeModelError(
		fmt.Sprintf("endpoints version %s, not supported", version), nil)
}

func decodeV3Endpoints(modelDef modelDefinition, opts DecodeModelOptions) (Resolver, error) {
	b, ok := modelDef["partitions"]
	if !ok {
		return nil, newDecodeModelError("endpoints model missing partitions", nil)
	}

	ps := partitions{}
	if err := json.Unmarshal(b, &ps); err != nil {
		return nil, newDecodeModelError("failed to decode endpoints model", err)
	}

	if opts.SkipCustomizations {
		return ps, nil
	}

	// Customization
	for i := 0; i < len(ps); i++ {
		p := &ps[i]
		custAddEC2Metadata(p)
		custAddS3DualStack(p)
		custRmIotDataService(p)
		custFixAppAutoscalingChina(p)
		custFixAppAutoscalingUsGov(p)
	}

	return ps, nil
}

func custAddS3DualStack(p *partition) {
	if p.ID != "aws" {
		return
	}

	custAddDualstack(p, "s3")
	custAddDualstack(p, "s3-control")
}

func custAddDualstack(p *partition, svcName string) {
	s, ok := p.Services[svcName]
	if !ok {
		return
	}

	s.Defaults.HasDualStack = boxedTrue
	s.Defaults.DualStackHostname = "{service}.dualstack.{region}.{dnsSuffix}"

	p.Services[svcName] = s
}

func custAddEC2Metadata(p *partition) {
	p.Services["ec2metadata"] = service{
		IsRegionalized:    boxedFalse,
		PartitionEndpoint: "aws-global",
		Endpoints: endpoints{
			"aws-global": endpoint{
				Hostname:  "169.254.169.254/latest",
				Protocols: []string{"http"},
			},
		},
	}
}

func custRmIotDataService(p *partition) {
	delete(p.Services, "data.iot")
}

func custFixAppAutoscalingChina(p *partition) {
	if p.ID != "aws-cn" {
		return
	}

	const serviceName = "application-autoscaling"
	s, ok := p.Services[serviceName]
	if !ok {
		return
	}

	const expectHostname = `autoscaling.{region}.amazonaws.com`
	if e, a := s.Defaults.Hostname, expectHostname; e != a {
		fmt.Printf("custFixAppAutoscalingChina: ignoring customization, expected %s, got %s\n", e, a)
		return
	}

	s.Defaults.Hostname = expectHostname + ".cn"
	p.Services[serviceName] = s
}

func custFixAppAutoscalingUsGov(p *partition) {
	if p.ID != "aws-us-gov" {
		return
	}

	const serviceName = "application-autoscaling"
	s, ok := p.Services[serviceName]
	if !ok {
		return
	}

	if a := s.Defaults.CredentialScope.Service; a != "" {
		fmt.Printf("custFixAppAutoscalingUsGov: ignoring customization, expected empty credential scope service, got %s\n", a)
		return
	}

	if a := s.Defaults.Hostname; a != "" {
		fmt.Printf("custFixAppAutoscalingUsGov: ignoring customization, expected empty hostname, got %s\n", a)
		return
	}

	s.Defaults.CredentialScope.Service = "application-autoscaling"
	s.Defaults.Hostname = "autoscaling.{region}.amazonaws.com"

	p.Services[serviceName] = s
}

type decodeModelError struct {
	awsError
}

func newDecodeModelError(msg string, err error) decodeModelError {
	return decodeModelError{
		awsError: awserr.New("DecodeEndpointsModelError", msg, err),
	}
}
