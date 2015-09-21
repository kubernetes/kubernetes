//
// goamz - Go packages to interact with the Amazon Web Services.
//
//   https://wiki.ubuntu.com/goamz
//
// Copyright (c) 2011 Canonical Ltd.
//
// Written by Gustavo Niemeyer <gustavo.niemeyer@canonical.com>
//
package aws

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/vaughan0/go-ini"
	"io/ioutil"
	"os"
)

// Region defines the URLs where AWS services may be accessed.
//
// See http://goo.gl/d8BP1 for more details.
type Region struct {
	Name                 string // the canonical name of this region.
	EC2Endpoint          string
	S3Endpoint           string
	S3BucketEndpoint     string // Not needed by AWS S3. Use ${bucket} for bucket name.
	S3LocationConstraint bool   // true if this region requires a LocationConstraint declaration.
	S3LowercaseBucket    bool   // true if the region requires bucket names to be lower case.
	SDBEndpoint          string
	SNSEndpoint          string
	SQSEndpoint          string
	IAMEndpoint          string
	ELBEndpoint          string
	AutoScalingEndpoint  string
	RdsEndpoint          string
	Route53Endpoint      string
}

var USGovWest = Region{
	"us-gov-west-1",
	"https://ec2.us-gov-west-1.amazonaws.com",
	"https://s3-fips-us-gov-west-1.amazonaws.com",
	"",
	true,
	true,
	"",
	"https://sns.us-gov-west-1.amazonaws.com",
	"https://sqs.us-gov-west-1.amazonaws.com",
	"https://iam.us-gov.amazonaws.com",
	"https://elasticloadbalancing.us-gov-west-1.amazonaws.com",
	"https://autoscaling.us-gov-west-1.amazonaws.com",
	"https://rds.us-gov-west-1.amazonaws.com",
	"https://route53.amazonaws.com",
}

var USEast = Region{
	"us-east-1",
	"https://ec2.us-east-1.amazonaws.com",
	"https://s3.amazonaws.com",
	"",
	false,
	false,
	"https://sdb.amazonaws.com",
	"https://sns.us-east-1.amazonaws.com",
	"https://sqs.us-east-1.amazonaws.com",
	"https://iam.amazonaws.com",
	"https://elasticloadbalancing.us-east-1.amazonaws.com",
	"https://autoscaling.us-east-1.amazonaws.com",
	"https://rds.us-east-1.amazonaws.com",
	"https://route53.amazonaws.com",
}

var USWest = Region{
	"us-west-1",
	"https://ec2.us-west-1.amazonaws.com",
	"https://s3-us-west-1.amazonaws.com",
	"",
	true,
	true,
	"https://sdb.us-west-1.amazonaws.com",
	"https://sns.us-west-1.amazonaws.com",
	"https://sqs.us-west-1.amazonaws.com",
	"https://iam.amazonaws.com",
	"https://elasticloadbalancing.us-west-1.amazonaws.com",
	"https://autoscaling.us-west-1.amazonaws.com",
	"https://rds.us-west-1.amazonaws.com",
	"https://route53.amazonaws.com",
}

var USWest2 = Region{
	"us-west-2",
	"https://ec2.us-west-2.amazonaws.com",
	"https://s3-us-west-2.amazonaws.com",
	"",
	true,
	true,
	"https://sdb.us-west-2.amazonaws.com",
	"https://sns.us-west-2.amazonaws.com",
	"https://sqs.us-west-2.amazonaws.com",
	"https://iam.amazonaws.com",
	"https://elasticloadbalancing.us-west-2.amazonaws.com",
	"https://autoscaling.us-west-2.amazonaws.com",
	"https://rds.us-west-2.amazonaws.com",
	"https://route53.amazonaws.com",
}

var EUWest = Region{
	"eu-west-1",
	"https://ec2.eu-west-1.amazonaws.com",
	"https://s3-eu-west-1.amazonaws.com",
	"",
	true,
	true,
	"https://sdb.eu-west-1.amazonaws.com",
	"https://sns.eu-west-1.amazonaws.com",
	"https://sqs.eu-west-1.amazonaws.com",
	"https://iam.amazonaws.com",
	"https://elasticloadbalancing.eu-west-1.amazonaws.com",
	"https://autoscaling.eu-west-1.amazonaws.com",
	"https://rds.eu-west-1.amazonaws.com",
	"https://route53.amazonaws.com",
}

var APSoutheast = Region{
	"ap-southeast-1",
	"https://ec2.ap-southeast-1.amazonaws.com",
	"https://s3-ap-southeast-1.amazonaws.com",
	"",
	true,
	true,
	"https://sdb.ap-southeast-1.amazonaws.com",
	"https://sns.ap-southeast-1.amazonaws.com",
	"https://sqs.ap-southeast-1.amazonaws.com",
	"https://iam.amazonaws.com",
	"https://elasticloadbalancing.ap-southeast-1.amazonaws.com",
	"https://autoscaling.ap-southeast-1.amazonaws.com",
	"https://rds.ap-southeast-1.amazonaws.com",
	"https://route53.amazonaws.com",
}

var APSoutheast2 = Region{
	"ap-southeast-2",
	"https://ec2.ap-southeast-2.amazonaws.com",
	"https://s3-ap-southeast-2.amazonaws.com",
	"",
	true,
	true,
	"https://sdb.ap-southeast-2.amazonaws.com",
	"https://sns.ap-southeast-2.amazonaws.com",
	"https://sqs.ap-southeast-2.amazonaws.com",
	"https://iam.amazonaws.com",
	"https://elasticloadbalancing.ap-southeast-2.amazonaws.com",
	"https://autoscaling.ap-southeast-2.amazonaws.com",
	"https://rds.ap-southeast-2.amazonaws.com",
	"https://route53.amazonaws.com",
}

var APNortheast = Region{
	"ap-northeast-1",
	"https://ec2.ap-northeast-1.amazonaws.com",
	"https://s3-ap-northeast-1.amazonaws.com",
	"",
	true,
	true,
	"https://sdb.ap-northeast-1.amazonaws.com",
	"https://sns.ap-northeast-1.amazonaws.com",
	"https://sqs.ap-northeast-1.amazonaws.com",
	"https://iam.amazonaws.com",
	"https://elasticloadbalancing.ap-northeast-1.amazonaws.com",
	"https://autoscaling.ap-northeast-1.amazonaws.com",
	"https://rds.ap-northeast-1.amazonaws.com",
	"https://route53.amazonaws.com",
}

var SAEast = Region{
	"sa-east-1",
	"https://ec2.sa-east-1.amazonaws.com",
	"https://s3-sa-east-1.amazonaws.com",
	"",
	true,
	true,
	"https://sdb.sa-east-1.amazonaws.com",
	"https://sns.sa-east-1.amazonaws.com",
	"https://sqs.sa-east-1.amazonaws.com",
	"https://iam.amazonaws.com",
	"https://elasticloadbalancing.sa-east-1.amazonaws.com",
	"https://autoscaling.sa-east-1.amazonaws.com",
	"https://rds.sa-east-1.amazonaws.com",
	"https://route53.amazonaws.com",
}

var CNNorth = Region{
	"cn-north-1",
	"https://ec2.cn-north-1.amazonaws.com.cn",
	"https://s3.cn-north-1.amazonaws.com.cn",
	"",
	true,
	true,
	"",
	"https://sns.cn-north-1.amazonaws.com.cn",
	"https://sqs.cn-north-1.amazonaws.com.cn",
	"https://iam.cn-north-1.amazonaws.com.cn",
	"https://elasticloadbalancing.cn-north-1.amazonaws.com.cn",
	"https://autoscaling.cn-north-1.amazonaws.com.cn",
	"https://rds.cn-north-1.amazonaws.com.cn",
	"https://route53.amazonaws.com",
}

var Regions = map[string]Region{
	APNortheast.Name:  APNortheast,
	APSoutheast.Name:  APSoutheast,
	APSoutheast2.Name: APSoutheast2,
	EUWest.Name:       EUWest,
	USEast.Name:       USEast,
	USWest.Name:       USWest,
	USWest2.Name:      USWest2,
	SAEast.Name:       SAEast,
	USGovWest.Name:    USGovWest,
	CNNorth.Name:      CNNorth,
}

type Auth struct {
	AccessKey, SecretKey, Token string
}

var unreserved = make([]bool, 128)
var hex = "0123456789ABCDEF"

func init() {
	// RFC3986
	u := "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz01234567890-_.~"
	for _, c := range u {
		unreserved[c] = true
	}
}

type credentials struct {
	Code            string
	LastUpdated     string
	Type            string
	AccessKeyId     string
	SecretAccessKey string
	Token           string
	Expiration      string
}

// GetMetaData retrieves instance metadata about the current machine.
//
// See http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AESDG-chapter-instancedata.html for more details.
func GetMetaData(path string) (contents []byte, err error) {
	url := "http://169.254.169.254/latest/meta-data/" + path

	resp, err := RetryingClient.Get(url)
	if err != nil {
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		err = fmt.Errorf("Code %d returned for url %s", resp.StatusCode, url)
		return
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return
	}
	return []byte(body), err
}

func getInstanceCredentials() (cred credentials, err error) {
	credentialPath := "iam/security-credentials/"

	// Get the instance role
	role, err := GetMetaData(credentialPath)
	if err != nil {
		return
	}

	// Get the instance role credentials
	credentialJSON, err := GetMetaData(credentialPath + string(role))
	if err != nil {
		return
	}

	err = json.Unmarshal([]byte(credentialJSON), &cred)
	return
}

// GetAuth creates an Auth based on either passed in credentials,
// environment information or instance based role credentials.
func GetAuth(accessKey string, secretKey string) (auth Auth, err error) {
	// First try passed in credentials
	if accessKey != "" && secretKey != "" {
		return Auth{accessKey, secretKey, ""}, nil
	}

	// Next try to get auth from the environment
	auth, err = SharedAuth()
	if err == nil {
		// Found auth, return
		return
	}

	// Next try to get auth from the environment
	auth, err = EnvAuth()
	if err == nil {
		// Found auth, return
		return
	}

	// Next try getting auth from the instance role
	cred, err := getInstanceCredentials()
	if err == nil {
		// Found auth, return
		auth.AccessKey = cred.AccessKeyId
		auth.SecretKey = cred.SecretAccessKey
		auth.Token = cred.Token
		return
	}
	err = errors.New("No valid AWS authentication found")
	return
}

// SharedAuth creates an Auth based on shared credentials stored in
// $HOME/.aws/credentials. The AWS_PROFILE environment variables is used to
// select the profile.
func SharedAuth() (auth Auth, err error) {
	var profileName = os.Getenv("AWS_PROFILE")

	if profileName == "" {
		profileName = "default"
	}

	var homeDir = os.Getenv("HOME")
	if homeDir == "" {
		err = errors.New("Could not get HOME")
		return
	}

	var credentialsFile = homeDir + "/.aws/credentials"
	file, err := ini.LoadFile(credentialsFile)
	if err != nil {
		err = errors.New("Couldn't parse AWS credentials file")
		return
	}

	var profile = file[profileName]
	if profile == nil {
		err = errors.New("Couldn't find profile in AWS credentials file")
		return
	}

	auth.AccessKey = profile["aws_access_key_id"]
	auth.SecretKey = profile["aws_secret_access_key"]

	if auth.AccessKey == "" {
		err = errors.New("AWS_ACCESS_KEY_ID not found in environment in credentials file")
	}
	if auth.SecretKey == "" {
		err = errors.New("AWS_SECRET_ACCESS_KEY not found in credentials file")
	}
	return
}

// EnvAuth creates an Auth based on environment information.
// The AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment
// For accounts that require a security token, it is read from AWS_SECURITY_TOKEN
// variables are used.
func EnvAuth() (auth Auth, err error) {
	auth.AccessKey = os.Getenv("AWS_ACCESS_KEY_ID")
	if auth.AccessKey == "" {
		auth.AccessKey = os.Getenv("AWS_ACCESS_KEY")
	}

	auth.SecretKey = os.Getenv("AWS_SECRET_ACCESS_KEY")
	if auth.SecretKey == "" {
		auth.SecretKey = os.Getenv("AWS_SECRET_KEY")
	}

	auth.Token = os.Getenv("AWS_SECURITY_TOKEN")

	if auth.AccessKey == "" {
		err = errors.New("AWS_ACCESS_KEY_ID or AWS_ACCESS_KEY not found in environment")
	}
	if auth.SecretKey == "" {
		err = errors.New("AWS_SECRET_ACCESS_KEY or AWS_SECRET_KEY not found in environment")
	}
	return
}

// Encode takes a string and URI-encodes it in a way suitable
// to be used in AWS signatures.
func Encode(s string) string {
	encode := false
	for i := 0; i != len(s); i++ {
		c := s[i]
		if c > 127 || !unreserved[c] {
			encode = true
			break
		}
	}
	if !encode {
		return s
	}
	e := make([]byte, len(s)*3)
	ei := 0
	for i := 0; i != len(s); i++ {
		c := s[i]
		if c > 127 || !unreserved[c] {
			e[ei] = '%'
			e[ei+1] = hex[c>>4]
			e[ei+2] = hex[c&0xF]
			ei += 3
		} else {
			e[ei] = c
			ei += 1
		}
	}
	return string(e[:ei])
}
