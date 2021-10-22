// Package endpoints contains the models for endpoints that should be used
// to generate endpoint definition files for the SDK.
package endpoints

//go:generate go run -tags codegen ../../private/model/cli/gen-endpoints/main.go -model ./endpoints.json -out ../../aws/endpoints/defaults.go
//go:generate gofmt -s -w ../../aws/endpoints
