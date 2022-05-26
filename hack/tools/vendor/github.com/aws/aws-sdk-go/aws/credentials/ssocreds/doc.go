// Package ssocreds provides a credential provider for retrieving temporary AWS credentials using an SSO access token.
//
// IMPORTANT: The provider in this package does not initiate or perform the AWS SSO login flow. The SDK provider
// expects that you have already performed the SSO login flow using AWS CLI using the "aws sso login" command, or by
// some other mechanism. The provider must find a valid non-expired access token for the AWS SSO user portal URL in
// ~/.aws/sso/cache. If a cached token is not found, it is expired, or the file is malformed an error will be returned.
//
// Loading AWS SSO credentials with the AWS shared configuration file
//
// You can use configure AWS SSO credentials from the AWS shared configuration file by
// providing the specifying the required keys in the profile:
//
//  sso_account_id
//  sso_region
//  sso_role_name
//  sso_start_url
//
// For example, the following defines a profile "devsso" and specifies the AWS SSO parameters that defines the target
// account, role, sign-on portal, and the region where the user portal is located. Note: all SSO arguments must be
// provided, or an error will be returned.
//
//  [profile devsso]
//  sso_start_url = https://my-sso-portal.awsapps.com/start
//  sso_role_name = SSOReadOnlyRole
//  sso_region = us-east-1
//  sso_account_id = 123456789012
//
// Using the config module, you can load the AWS SDK shared configuration, and specify that this profile be used to
// retrieve credentials. For example:
//
//  sess, err := session.NewSessionWithOptions(session.Options{
//      SharedConfigState: session.SharedConfigEnable,
//      Profile:           "devsso",
//  })
//  if err != nil {
//      return err
//  }
//
// Programmatically loading AWS SSO credentials directly
//
// You can programmatically construct the AWS SSO Provider in your application, and provide the necessary information
// to load and retrieve temporary credentials using an access token from ~/.aws/sso/cache.
//
//  svc := sso.New(sess, &aws.Config{
//      Region: aws.String("us-west-2"), // Client Region must correspond to the AWS SSO user portal region
//  })
//
//  provider := ssocreds.NewCredentialsWithClient(svc, "123456789012", "SSOReadOnlyRole", "https://my-sso-portal.awsapps.com/start")
//
//  credentials, err := provider.Get()
//  if err != nil {
//      return err
//  }
//
// Additional Resources
//
// Configuring the AWS CLI to use AWS Single Sign-On: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html
//
// AWS Single Sign-On User Guide: https://docs.aws.amazon.com/singlesignon/latest/userguide/what-is.html
package ssocreds
