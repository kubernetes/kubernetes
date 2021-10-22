// +build example

package main

import (
	"fmt"
	"os"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ecr"
)

const DEFAULT_AWS_REGION = "us-east-1"

// This example deletes an ECR Repository
//
// Usage:
// AWS_REGION=us-east-1 go run -tags example deleteECRRepository.go <repo_name>
func main() {

	config := &aws.Config{Region: aws.String(getAwsRegion())}
	svc := ecr.New(session.New(), config)

	repoName := getRepoNameArg()
	if repoName == "" {
		printUsageAndExit1()
	}
	input := &ecr.DeleteRepositoryInput{
		Force:          aws.Bool(false),
		RepositoryName: aws.String(repoName),
	}

	output, err := svc.DeleteRepository(input)
	if err != nil {
		fmt.Printf("\nError deleting the repo %v in region %v\n%v\n", repoName, aws.StringValue(config.Region), err.Error())
		os.Exit(1)
	}

	fmt.Printf("\nECR Repository \"%v\" deleted successfully!\n\nAWS Output:\n%v", repoName, output)
}

// Print correct usage and exit the program with code 1
func printUsageAndExit1() {
	fmt.Println("\nUsage: AWS_REGION=us-east-1 go run -tags example deleteECRRepository.go <repo_name>")
	os.Exit(1)
}

// Try get the repo name from the first argument
func getRepoNameArg() string {
	if len(os.Args) < 2 {
		return ""
	}
	firstArg := os.Args[1]
	return firstArg
}

// Returns the aws region from env var or default region defined in DEFAULT_AWS_REGION constant
func getAwsRegion() string {
	awsRegion := os.Getenv("AWS_REGION")
	if awsRegion != "" {
		return awsRegion
	}
	return DEFAULT_AWS_REGION
}
