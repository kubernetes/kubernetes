/*
Package session provides configuration for the SDK's service clients.

Sessions can be shared across all service clients that share the same base
configuration.  The Session is built from the SDK's default configuration and
request handlers.

Sessions should be cached when possible, because creating a new Session will
load all configuration values from the environment, and config files each time
the Session is created. Sharing the Session value across all of your service
clients will ensure the configuration is loaded the fewest number of times possible.

Concurrency

Sessions are safe to use concurrently as long as the Session is not being
modified. The SDK will not modify the Session once the Session has been created.
Creating service clients concurrently from a shared Session is safe.

Sessions from Shared Config

Sessions can be created using the method above that will only load the
additional config if the AWS_SDK_LOAD_CONFIG environment variable is set.
Alternatively you can explicitly create a Session with shared config enabled.
To do this you can use NewSessionWithOptions to configure how the Session will
be created. Using the NewSessionWithOptions with SharedConfigState set to
SharedConfigEnabled will create the session as if the AWS_SDK_LOAD_CONFIG
environment variable was set.

Creating Sessions

When creating Sessions optional aws.Config values can be passed in that will
override the default, or loaded config values the Session is being created
with. This allows you to provide additional, or case based, configuration
as needed.

By default NewSession will only load credentials from the shared credentials
file (~/.aws/credentials). If the AWS_SDK_LOAD_CONFIG environment variable is
set to a truthy value the Session will be created from the configuration
values from the shared config (~/.aws/config) and shared credentials
(~/.aws/credentials) files. See the section Sessions from Shared Config for
more information.

Create a Session with the default config and request handlers. With credentials
region, and profile loaded from the environment and shared config automatically.
Requires the AWS_PROFILE to be set, or "default" is used.

	// Create Session
	sess, err := session.NewSession()

	// Create a Session with a custom region
	sess, err := session.NewSession(&aws.Config{Region: aws.String("us-east-1")})

	// Create a S3 client instance from a session
	sess, err := session.NewSession()
	if err != nil {
		// Handle Session creation error
	}
	svc := s3.New(sess)

Create Session With Option Overrides

In addition to NewSession, Sessions can be created using NewSessionWithOptions.
This func allows you to control and override how the Session will be created
through code instead of being driven by environment variables only.

Use NewSessionWithOptions when you want to provide the config profile, or
override the shared config state (AWS_SDK_LOAD_CONFIG).

	// Equivalent to session.NewSession()
	sess, err := session.NewSessionWithOptions(session.Options{})

	// Specify profile to load for the session's config
	sess, err := session.NewSessionWithOptions(session.Options{
		 Profile: "profile_name",
	})

	// Specify profile for config and region for requests
	sess, err := session.NewSessionWithOptions(session.Options{
		 Config: aws.Config{Region: aws.String("us-east-1")},
		 Profile: "profile_name",
	})

	// Force enable Shared Config support
	sess, err := session.NewSessionWithOptions(session.Options{
		SharedConfigState: SharedConfigEnable,
	})

Adding Handlers

You can add handlers to a session for processing HTTP requests. All service
clients that use the session inherit the handlers. For example, the following
handler logs every request and its payload made by a service client:

	// Create a session, and add additional handlers for all service
	// clients created with the Session to inherit. Adds logging handler.
	sess, err := session.NewSession()
	sess.Handlers.Send.PushFront(func(r *request.Request) {
		// Log every request made and its payload
		logger.Println("Request: %s/%s, Payload: %s",
			r.ClientInfo.ServiceName, r.Operation, r.Params)
	})

Deprecated "New" function

The New session function has been deprecated because it does not provide good
way to return errors that occur when loading the configuration files and values.
Because of this, NewSession was created so errors can be retrieved when
creating a session fails.

Shared Config Fields

By default the SDK will only load the shared credentials file's (~/.aws/credentials)
credentials values, and all other config is provided by the environment variables,
SDK defaults, and user provided aws.Config values.

If the AWS_SDK_LOAD_CONFIG environment variable is set, or SharedConfigEnable
option is used to create the Session the full shared config values will be
loaded. This includes credentials, region, and support for assume role. In
addition the Session will load its configuration from both the shared config
file (~/.aws/config) and shared credentials file (~/.aws/credentials). Both
files have the same format.

If both config files are present the configuration from both files will be
read. The Session will be created from  configuration values from the shared
credentials file (~/.aws/credentials) over those in the shared credentials
file (~/.aws/config).

Credentials are the values the SDK should use for authenticating requests with
AWS Services. They arfrom a configuration file will need to include both
aws_access_key_id and aws_secret_access_key must be provided together in the
same file to be considered valid. The values will be ignored if not a complete
group. aws_session_token is an optional field that can be provided if both of
the other two fields are also provided.

	aws_access_key_id = AKID
	aws_secret_access_key = SECRET
	aws_session_token = TOKEN

Assume Role values allow you to configure the SDK to assume an IAM role using
a set of credentials provided in a config file via the source_profile field.
Both "role_arn" and "source_profile" are required. The SDK does not support
assuming a role with MFA token Via the Session's constructor. You can use the
stscreds.AssumeRoleProvider credentials provider to specify custom
configuration and support for MFA.

	role_arn = arn:aws:iam::<account_number>:role/<role_name>
	source_profile = profile_with_creds
	external_id = 1234
	mfa_serial = not supported!
	role_session_name = session_name

Region is the region the SDK should use for looking up AWS service endpoints
and signing requests.

	region = us-east-1

Environment Variables

When a Session is created several environment variables can be set to adjust
how the SDK functions, and what configuration data it loads when creating
Sessions. All environment values are optional, but some values like credentials
require multiple of the values to set or the partial values will be ignored.
All environment variable values are strings unless otherwise noted.

Environment configuration values. If set both Access Key ID and Secret Access
Key must be provided. Session Token and optionally also be provided, but is
not required.

	# Access Key ID
	AWS_ACCESS_KEY_ID=AKID
	AWS_ACCESS_KEY=AKID # only read if AWS_ACCESS_KEY_ID is not set.

	# Secret Access Key
	AWS_SECRET_ACCESS_KEY=SECRET
	AWS_SECRET_KEY=SECRET=SECRET # only read if AWS_SECRET_ACCESS_KEY is not set.

	# Session Token
	AWS_SESSION_TOKEN=TOKEN

Region value will instruct the SDK where to make service API requests to. If is
not provided in the environment the region must be provided before a service
client request is made.

	AWS_REGION=us-east-1

	# AWS_DEFAULT_REGION is only read if AWS_SDK_LOAD_CONFIG is also set,
	# and AWS_REGION is not also set.
	AWS_DEFAULT_REGION=us-east-1

Profile name the SDK should load use when loading shared config from the
configuration files. If not provided "default" will be used as the profile name.

	AWS_PROFILE=my_profile

	# AWS_DEFAULT_PROFILE is only read if AWS_SDK_LOAD_CONFIG is also set,
	# and AWS_PROFILE is not also set.
	AWS_DEFAULT_PROFILE=my_profile

SDK load config instructs the SDK to load the shared config in addition to
shared credentials. This also expands the configuration loaded so the shared
credentials will have parity with the shared config file. This also enables
Region and Profile support for the AWS_DEFAULT_REGION and AWS_DEFAULT_PROFILE
env values as well.

	AWS_SDK_LOAD_CONFIG=1

Shared credentials file path can be set to instruct the SDK to use an alternative
file for the shared credentials. If not set the file will be loaded from
$HOME/.aws/credentials on Linux/Unix based systems, and
%USERPROFILE%\.aws\credentials on Windows.

	AWS_SHARED_CREDENTIALS_FILE=$HOME/my_shared_credentials

Shared config file path can be set to instruct the SDK to use an alternative
file for the shared config. If not set the file will be loaded from
$HOME/.aws/config on Linux/Unix based systems, and
%USERPROFILE%\.aws\config on Windows.

	AWS_CONFIG_FILE=$HOME/my_shared_config


*/
package session
