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
//	resolver, err := endpoints.DecodeModel(reader)
//
//	partitions := resolver.(endpoints.EnumPartitions).Partitions()
//	for _, p := range partitions {
//	    // ... inspect partitions
//	}
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
		custRegionalS3(p)
		custRmIotDataService(p)
		custFixAppAutoscalingChina(p)
		custFixAppAutoscalingUsGov(p)
	}

	return ps, nil
}

func custRegionalS3(p *partition) {
	if p.ID != "aws" {
		return
	}

	service, ok := p.Services["s3"]
	if !ok {
		return
	}

	const awsGlobal = "aws-global"
	const usEast1 = "us-east-1"

	// If global endpoint already exists no customization needed.
	if _, ok := service.Endpoints[endpointKey{Region: awsGlobal}]; ok {
		return
	}

	service.PartitionEndpoint = awsGlobal
	if _, ok := service.Endpoints[endpointKey{Region: usEast1}]; !ok {
		service.Endpoints[endpointKey{Region: usEast1}] = endpoint{}
	}
	service.Endpoints[endpointKey{Region: awsGlobal}] = endpoint{
		Hostname: "s3.amazonaws.com",
		CredentialScope: credentialScope{
			Region: usEast1,
		},
	}

	p.Services["s3"] = service
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
	serviceDefault := s.Defaults[defaultKey{}]
	if e, a := expectHostname, serviceDefault.Hostname; e != a {
		fmt.Printf("custFixAppAutoscalingChina: ignoring customization, expected %s, got %s\n", e, a)
		return
	}
	serviceDefault.Hostname = expectHostname + ".cn"
	s.Defaults[defaultKey{}] = serviceDefault
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

	serviceDefault := s.Defaults[defaultKey{}]
	if a := serviceDefault.CredentialScope.Service; a != "" {
		fmt.Printf("custFixAppAutoscalingUsGov: ignoring customization, expected empty credential scope service, got %s\n", a)
		return
	}

	if a := serviceDefault.Hostname; a != "" {
		fmt.Printf("custFixAppAutoscalingUsGov: ignoring customization, expected empty hostname, got %s\n", a)
		return
	}

	serviceDefault.CredentialScope.Service = "application-autoscaling"
	serviceDefault.Hostname = "autoscaling.{region}.amazonaws.com"

	if s.Defaults == nil {
		s.Defaults = make(endpointDefaults)
	}

	s.Defaults[defaultKey{}] = serviceDefault

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
