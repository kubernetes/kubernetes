# Amazon Bedrock AgentCore Construct Library

| **Language**                                                                                   | **Package**                             |
| :--------------------------------------------------------------------------------------------- | --------------------------------------- |
| ![Typescript Logo](https://docs.aws.amazon.com/cdk/api/latest/img/typescript32.png) TypeScript | `aws-cdk-lib/aws-bedrockagentcore` |

[Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/) enables you to deploy and operate highly capable AI agents securely, at scale. It offers infrastructure purpose-built for dynamic agent workloads, powerful tools to enhance agents, and essential controls for real-world deployment. AgentCore services can be used together or independently and work with any framework including CrewAI, LangGraph, LlamaIndex, and Strands Agents, as well as any foundation model in or outside of Amazon Bedrock, giving you ultimate flexibility. AgentCore eliminates the undifferentiated heavy lifting of building specialized agent infrastructure, so you can accelerate agents to production.

This construct library facilitates the deployment of Bedrock AgentCore primitives, enabling you to create sophisticated AI applications that can interact with your systems and data sources.

> **Note:** Users need to ensure their CDK deployment role has the `iam:CreateServiceLinkedRole` permission for AgentCore service-linked roles.

## Table of contents

- [Amazon Bedrock AgentCore Construct Library](#amazon-bedrock-agentcore-construct-library)
  - [Table of contents](#table-of-contents)
  - [AgentCore Runtime](#agentcore-runtime)
    - [Runtime Endpoints](#runtime-endpoints)
    - [AgentCore Runtime Properties](#agentcore-runtime-properties)
    - [Runtime Endpoint Properties](#runtime-endpoint-properties)
    - [Creating a Runtime](#creating-a-runtime)
      - [Option 1: Use an existing image in ECR](#option-1-use-an-existing-image-in-ecr)
      - [Option 2: Use a local asset](#option-2-use-a-local-asset)
      - [Option 3: Use direct code deployment](#option-3-use-direct-code-deployment)
      - [Option 4: Use an ECR container image URI](#option-4-use-an-ecr-container-image-uri)
    - [Granting Permissions to Invoke Bedrock Models or Inference Profiles](#granting-permissions-to-invoke-bedrock-models-or-inference-profiles)
    - [Runtime Versioning](#runtime-versioning)
      - [Managing Endpoints and Versions](#managing-endpoints-and-versions)
        - [Step 1: Initial Deployment](#step-1-initial-deployment)
        - [Step 2: Creating Custom Endpoints](#step-2-creating-custom-endpoints)
        - [Step 3: Runtime Update Deployment](#step-3-runtime-update-deployment)
        - [Step 4: Testing with Staging Endpoints](#step-4-testing-with-staging-endpoints)
        - [Step 5: Promoting to Production](#step-5-promoting-to-production)
    - [Creating Standalone Runtime Endpoints](#creating-standalone-runtime-endpoints)
      - [Example: Creating an endpoint for an existing runtime](#example-creating-an-endpoint-for-an-existing-runtime)
    - [Runtime Authentication Configuration](#runtime-authentication-configuration)
      - [IAM Authentication (Default)](#iam-authentication-default)
      - [Cognito Authentication](#cognito-authentication)
      - [JWT Authentication](#jwt-authentication)
      - [OAuth Authentication](#oauth-authentication)
      - [Using a Custom IAM Role](#using-a-custom-iam-role)
    - [Runtime Network Configuration](#runtime-network-configuration)
      - [Public Network Mode (Default)](#public-network-mode-default)
      - [VPC Network Mode](#vpc-network-mode)
      - [Managing Security Groups with VPC Configuration](#managing-security-groups-with-vpc-configuration)
    - [Runtime IAM Permissions](#runtime-iam-permissions)
    - [Other configuration](#other-configuration)
      - [Lifecycle configuration](#lifecycle-configuration)
      - [Request header configuration](#request-header-configuration)
      - [Application log group](#application-log-group)
  - [Browser](#browser)
    - [Browser Network modes](#browser-network-modes)
    - [Browser Properties](#browser-properties)
    - [Basic Browser Creation](#basic-browser-creation)
    - [Browser with Tags](#browser-with-tags)
    - [Browser with VPC](#browser-with-vpc)
    - [Browser with Recording Configuration](#browser-with-recording-configuration)
    - [Browser with Custom Execution Role](#browser-with-custom-execution-role)
    - [Browser with S3 Recording and Permissions](#browser-with-s3-recording-and-permissions)
    - [Browser with Browser signing](#browser-with-browser-signing)
    - [Browser IAM Permissions](#browser-iam-permissions)
  - [Code Interpreter](#code-interpreter)
    - [Code Interpreter Network Modes](#code-interpreter-network-modes)
    - [Code Interpreter Properties](#code-interpreter-properties)
    - [Basic Code Interpreter Creation](#basic-code-interpreter-creation)
    - [Code Interpreter with VPC](#code-interpreter-with-vpc)
    - [Code Interpreter with Sandbox Network Mode](#code-interpreter-with-sandbox-network-mode)
    - [Code Interpreter with Custom Execution Role](#code-interpreter-with-custom-execution-role)
    - [Code Interpreter IAM Permissions](#code-interpreter-iam-permissions)
    - [Code interpreter with tags](#code-interpreter-with-tags)
  - [Gateway](#gateway)
    - [Gateway Properties](#gateway-properties)
    - [Basic Gateway Creation](#basic-gateway-creation)
    - [Protocol configuration](#protocol-configuration)
    - [Inbound authorization](#inbound-authorization)
    - [Gateway with KMS Encryption](#gateway-with-kms-encryption)
    - [Gateway with Custom Execution Role](#gateway-with-custom-execution-role)
    - [Gateway IAM Permissions](#gateway-iam-permissions)
  - [Gateway Target](#gateway-target)
    - [Gateway Target Properties](#gateway-target-properties)
    - [Targets types](#targets-types)
    - [Understanding Tool Naming](#understanding-tool-naming)
    - [Tools schema For Lambda target](#tools-schema-for-lambda-target)
    - [Api schema For OpenAPI and Smithy target](#api-schema-for-openapi-and-smithy-target)
    - [Outbound auth](#outbound-auth)
      - [Token Vault credential providers](#token-vault-credential-providers)
      - [Workload identities](#workload-identities)
    - [Basic Gateway Target Creation](#basic-gateway-target-creation)
      - [Using addTarget methods (Recommended)](#using-addtarget-methods-recommended)
      - [Using static factory methods](#using-static-factory-methods)
    - [Advanced Usage: Direct Configuration for gateway target](#advanced-usage-direct-configuration-for-gateway-target)
      - [Configuration Factory Methods](#configuration-factory-methods)
      - [Example: Lambda Target with Custom Configuration](#example-lambda-target-with-custom-configuration)
    - [Gateway Target IAM Permissions](#gateway-target-iam-permissions)
  - [Memory](#memory)
    - [Memory Properties](#memory-properties)
    - [Basic Memory Creation](#basic-memory-creation)
    - [LTM Memory Extraction Stategies](#ltm-memory-extraction-stategies)
    - [Memory with Built-in Strategies](#memory-with-built-in-strategies)
    - [Memory with custom Strategies](#memory-with-custom-strategies)
      - [Memory with Custom Execution Role](#memory-with-custom-execution-role)
    - [Memory with self-managed Strategies](#memory-with-self-managed-strategies)
    - [Memory Strategy Methods](#memory-strategy-methods)
  - [Online Evaluation](#online-evaluation)
    - [Online Evaluation Properties](#online-evaluation-properties)
    - [Basic Online Evaluation Creation](#basic-online-evaluation-creation)
    - [Built-in Evaluators](#built-in-evaluators)
    - [Custom Evaluators](#custom-evaluators)
      - [LLM-as-a-Judge Evaluator](#llm-as-a-judge-evaluator)
      - [Code-Based Evaluator](#code-based-evaluator)
      - [Using Custom Evaluators with Online Evaluation](#using-custom-evaluators-with-online-evaluation)
    - [Data Source Configuration](#data-source-configuration)
    - [Sampling and Filtering](#sampling-and-filtering)
    - [Online Evaluation with Custom Execution Role](#online-evaluation-with-custom-execution-role)
    - [Online Evaluation IAM Permissions](#online-evaluation-iam-permissions)

## AgentCore Runtime

The AgentCore Runtime construct enables you to deploy containerized agents on Amazon Bedrock AgentCore.
This L2 construct simplifies runtime creation just pass your ECR repository name
and the construct handles all the configuration with sensible defaults.

### Runtime Endpoints

Endpoints provide a stable way to invoke specific versions of your agent runtime, enabling controlled deployments across different environments.
When you create an agent runtime, Amazon Bedrock AgentCore automatically creates a "DEFAULT" endpoint which always points to the latest version
of runtime.

You can create additional endpoints in two ways:

1. **Using Runtime.addEndpoint()** - Convenient method when creating endpoints alongside the runtime.
2. **Using RuntimeEndpoint** - Flexible approach for existing runtimes.

For example, you might keep a "production" endpoint on a stable version while testing newer versions
through a "staging" endpoint. This separation allows you to test changes thoroughly before promoting them
to production by simply updating the endpoint to point to the newer version.

### AgentCore Runtime Properties

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `runtimeName` | `string` | No | The name of the agent runtime. Valid characters are a-z, A-Z, 0-9, _ (underscore). Must start with a letter and can be up to 48 characters long. If not provided, a unique name will be auto-generated |
| `agentRuntimeArtifact` | `AgentRuntimeArtifact` | Yes | The artifact configuration for the agent runtime containing the container configuration with ECR URI |
| `executionRole` | `iam.IRole` | No | The IAM role that provides permissions for the agent runtime. If not provided, a role will be created automatically |
| `networkConfiguration` | `NetworkConfiguration` | No | Network configuration for the agent runtime. Defaults to `RuntimeNetworkConfiguration.usingPublicNetwork()` |
| `description` | `string` | No | Optional description for the agent runtime |
| `protocolConfiguration` | `ProtocolType` | No | Protocol configuration for the agent runtime. Defaults to `ProtocolType.HTTP` |
| `authorizerConfiguration` | `RuntimeAuthorizerConfiguration` | No | Authorizer configuration for the agent runtime. Use `RuntimeAuthorizerConfiguration` static methods to create configurations for IAM, Cognito, JWT, or OAuth authentication |
| `environmentVariables` | `{ [key: string]: string }` | No | Environment variables for the agent runtime. Maximum 50 environment variables |
| `tags` | `{ [key: string]: string }` | No | Tags for the agent runtime. A list of key:value pairs of tags to apply to this Runtime resource |
| `lifecycleConfiguration` | LifecycleConfiguration | No | The life cycle configuration for the AgentCore Runtime. Defaults to 900 seconds (15 minutes) for idle, 28800 seconds (8 hours) for max life time |
| `requestHeaderConfiguration` | RequestHeaderConfiguration | No | Configuration for HTTP request headers that will be passed through to the runtime. Defaults to no configuration |

### Runtime Endpoint Properties

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `endpointName` | `string` | No | The name of the runtime endpoint. Valid characters are a-z, A-Z, 0-9, _ (underscore). Must start with a letter and can be up to 48 characters long. If not provided, a unique name will be auto-generated |
| `agentRuntimeId` | `string` | Yes | The Agent Runtime ID for this endpoint |
| `agentRuntimeVersion` | `string` | Yes | The Agent Runtime version for this endpoint. Must be between 1 and 5 characters long.|
| `description` | `string` | No | Optional description for the runtime endpoint |
| `tags` | `{ [key: string]: string }` | No | Tags for the runtime endpoint |

### Creating a Runtime

#### Option 1: Use an existing image in ECR

Reference an image available within ECR.

```typescript fixture=default
const repository = new ecr.Repository(this, "TestRepository", {
  repositoryName: "test-agent-runtime",
});

// The runtime by default create ECR permission only for the repository available in the account the stack is being deployed
const agentRuntimeArtifact = agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, "v1.0.0");

// Create runtime using the built image
const runtime = new agentcore.Runtime(this, "MyAgentRuntime", {
  runtimeName: "myAgent",
  agentRuntimeArtifact: agentRuntimeArtifact
});
```

#### Option 2: Use a local asset

Reference a local directory containing a Dockerfile.
Images are built from a local Docker context directory (with a Dockerfile), uploaded to Amazon Elastic Container Registry (ECR)
by the CDK toolkit,and can be naturally referenced in your CDK app.

```typescript fixture=default
const agentRuntimeArtifact = agentcore.AgentRuntimeArtifact.fromAsset(
  path.join(__dirname, "path to agent dockerfile directory")
);

const runtime = new agentcore.Runtime(this, "MyAgentRuntime", {
  runtimeName: "myAgent",
  agentRuntimeArtifact: agentRuntimeArtifact,
});
```

#### Option 3: Use direct code deployment

With the container deployment method, developers create a Dockerfile, build ARM-compatible containers, manage ECR repositories, and upload containers for code changes. This works well where container DevOps pipelines have already been established to automate deployments.

However, customers looking for fully managed deployments can benefit from direct code deployment, which can significantly improve developer time and productivity. Direct code deployment provides a secure and scalable path forward for rapid prototyping agent capabilities to deploying production workloads at scale.

With direct code deployment, developers create a zip archive of code and dependencies, upload to Amazon S3, and configure the bucket in the agent configuration. A ZIP archive containing Linux arm64 dependencies needs to be uploaded to S3 as a pre-requisite to Create Agent Runtime.

For more information, please refer to the [documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-get-started-code-deploy.html).

```typescript fixture=default
// S3 bucket containing the agent core
const codeBucket = new s3.Bucket(this, "AgentCode", {
  bucketName: "my-code-bucket",
  removalPolicy: RemovalPolicy.DESTROY, // For demo purposes
});

// the bucket above needs to contain the agent code

const agentRuntimeArtifact = agentcore.AgentRuntimeArtifact.fromS3(
  {
    bucketName: codeBucket.bucketName,
    objectKey: 'deployment_package.zip',
  }, 
  agentcore.AgentCoreRuntime.PYTHON_3_12, 
  ['opentelemetry-instrument', 'main.py']
);

const runtimeInstance = new agentcore.Runtime(this, "MyAgentRuntime", {
  runtimeName: "myAgent",
  agentRuntimeArtifact: agentRuntimeArtifact,
});
```

Alternatively, you can use local code assets that will be automatically packaged and uploaded to a CDK-managed S3 bucket:

```typescript fixture=default
const agentRuntimeArtifact = agentcore.AgentRuntimeArtifact.fromCodeAsset({
  path: path.join(__dirname, 'path/to/agent/code'),
  runtime: agentcore.AgentCoreRuntime.PYTHON_3_12,
  entrypoint: ['opentelemetry-instrument', 'main.py'],
});

const runtimeInstance = new agentcore.Runtime(this, "MyAgentRuntime", {
  runtimeName: "myAgent",
  agentRuntimeArtifact: agentRuntimeArtifact,
});
```

#### Option 4: Use an ECR container image URI

Reference an ECR container image directly by its URI. This is useful when you have a pre-existing ECR image URI from CloudFormation parameters or cross-stack references. No IAM permissions are automatically granted - you must ensure the runtime has ECR pull permissions.

```typescript fixture=default
// Direct URI reference
const agentRuntimeArtifact = agentcore.AgentRuntimeArtifact.fromImageUri(
  "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-agent:v1.0.0"
);

const runtime = new agentcore.Runtime(this, "MyAgentRuntime", {
  runtimeName: "myAgent",
  agentRuntimeArtifact: agentRuntimeArtifact,
});
```

You can also use CloudFormation parameters or references:

```typescript fixture=default
// Using a CloudFormation parameter
const imageUriParam = new cdk.CfnParameter(this, "ImageUri", {
  type: "String",
  description: "Container image URI for the agent runtime",
});

const agentRuntimeArtifact = agentcore.AgentRuntimeArtifact.fromImageUri(
  imageUriParam.valueAsString
);

const runtime = new agentcore.Runtime(this, "MyAgentRuntime", {
  runtimeName: "myAgent",
  agentRuntimeArtifact: agentRuntimeArtifact,
});
```

### Granting Permissions to Invoke Bedrock Models or Inference Profiles

To grant the runtime permissions to invoke Bedrock models or inference profiles:

```typescript fixture=default
// Note: This example uses aws-cdk-lib/aws-bedrock for foundation model references
declare const runtime: agentcore.Runtime;

// Define the Bedrock Foundation Model
const model = bedrock.FoundationModel.fromFoundationModelId(this, 'Model',
  bedrock.FoundationModelIdentifier.ANTHROPIC_CLAUDE_3_7_SONNET_20250219_V1_0,
);

// Grant the runtime permissions to invoke the model
runtime.role.addToPrincipalPolicy(new iam.PolicyStatement({
  actions: ['bedrock:InvokeModel'],
  resources: [model.modelArn],
}));
```

### Runtime Versioning

Amazon Bedrock AgentCore automatically manages runtime versioning to ensure safe deployments and rollback capabilities.
When you create an agent runtime, AgentCore automatically creates version 1 (V1). Each subsequent update to the
runtime configuration (such as updating the container image, modifying network settings, or changing protocol configurations)
creates a new immutable version. These versions contain complete, self-contained configurations that can be referenced by endpoints,
allowing you to maintain different versions for different environments or gradually roll out updates.

#### Managing Endpoints and Versions

Amazon Bedrock AgentCore automatically manages runtime versioning to provide safe deployments and rollback capabilities. You can follow
the steps below to understand how to use versioning with runtime for controlled deployments across different environments.

##### Step 1: Initial Deployment

When you first create an agent runtime, AgentCore automatically creates Version 1 of your runtime. At this point, a DEFAULT endpoint is
automatically created that points to Version 1. This DEFAULT endpoint serves as the main access point for your runtime.

```typescript fixture=default
const repository = new ecr.Repository(this, "TestRepository", {
  repositoryName: "test-agent-runtime",
});

const runtime = new agentcore.Runtime(this, "MyAgentRuntime", {
  runtimeName: "myAgent",
  agentRuntimeArtifact: agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, "v1.0.0"),
});
```

##### Step 2: Creating Custom Endpoints

After the initial deployment, you can create additional endpoints for different environments. For example, you might create a "production"
endpoint that explicitly points to Version 1. This allows you to maintain stable access points for specific environments while keeping the
flexibility to test newer versions elsewhere.

```typescript fixture=default
const repository = new ecr.Repository(this, "TestRepository", {
  repositoryName: "test-agent-runtime",
});

const runtime = new agentcore.Runtime(this, "MyAgentRuntime", {
  runtimeName: "myAgent",
  agentRuntimeArtifact: agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, "v1.0.0"),
});

const prodEndpoint = runtime.addEndpoint("production", {
  version: "1",
  description: "Stable production endpoint - pinned to v1"
});
```

##### Step 3: Runtime Update Deployment

When you update the runtime configuration (such as updating the container image, modifying network settings, or changing protocol
configurations), AgentCore automatically creates a new version (Version 2). Upon this update:

- Version 2 is created automatically with the new configuration
- The DEFAULT endpoint automatically updates to point to Version 2
- Any explicitly pinned endpoints (like the production endpoint) remain on their specified versions

```typescript fixture=default
const repository = new ecr.Repository(this, "TestRepository", {
  repositoryName: "test-agent-runtime",
});

const agentRuntimeArtifactNew = agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, "v2.0.0");

const runtime = new agentcore.Runtime(this, "MyAgentRuntime", {
  runtimeName: "myAgent",
  agentRuntimeArtifact: agentRuntimeArtifactNew,
});
```

##### Step 4: Testing with Staging Endpoints

Once Version 2 exists, you can create a staging endpoint that points to the new version. This staging endpoint allows you to test the
new version in a controlled environment before promoting it to production. This separation ensures that production traffic continues
to use the stable version while you validate the new version.

```typescript fixture=default
const repository = new ecr.Repository(this, "TestRepository", {
  repositoryName: "test-agent-runtime",
});

const agentRuntimeArtifactNew = agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, "v2.0.0");

const runtime = new agentcore.Runtime(this, "MyAgentRuntime", {
  runtimeName: "myAgent",
  agentRuntimeArtifact: agentRuntimeArtifactNew,
});

const stagingEndpoint = runtime.addEndpoint("staging", {
  version: "2",
  description: "Staging environment for testing new version"
});
```

##### Step 5: Promoting to Production

After thoroughly testing the new version through the staging endpoint, you can update the production endpoint to point to Version 2.
This controlled promotion process ensures that you can validate changes before they affect production traffic.

```typescript fixture=default
const repository = new ecr.Repository(this, "TestRepository", {
  repositoryName: "test-agent-runtime",
});

const agentRuntimeArtifactNew = agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, "v2.0.0");

const runtime = new agentcore.Runtime(this, "MyAgentRuntime", {
  runtimeName: "myAgent",
  agentRuntimeArtifact: agentRuntimeArtifactNew,
});

const prodEndpoint = runtime.addEndpoint("production", {
  version: "2",  // New version added here
  description: "Stable production endpoint"
});
```

### Creating Standalone Runtime Endpoints

RuntimeEndpoint can also be created as a standalone resource.

#### Example: Creating an endpoint for an existing runtime

```typescript fixture=default
// Reference an existing runtime by its ID
const existingRuntimeId = "abc123-runtime-id"; // The ID of an existing runtime

// Create a standalone endpoint
const endpoint = new agentcore.RuntimeEndpoint(this, "MyEndpoint", {
  endpointName: "production",
  agentRuntimeId: existingRuntimeId,
  agentRuntimeVersion: "1", // Specify which version to use
  description: "Production endpoint for existing runtime"
});
```

### Runtime Authentication Configuration

The AgentCore Runtime supports multiple authentication modes to secure access to your agent endpoints. Authentication is configured during runtime creation using the `RuntimeAuthorizerConfiguration` class's static factory methods.

#### IAM Authentication (Default)

IAM authentication is the default mode, when no authorizerConfiguration is set then the underlying service use IAM.

#### Cognito Authentication

To configure AWS Cognito User Pool authentication:

```typescript fixture=default
declare const userPool: cognito.UserPool;
declare const userPoolClient: cognito.UserPoolClient;
declare const anotherUserPoolClient: cognito.UserPoolClient;

const repository = new ecr.Repository(this, "TestRepository", {
  repositoryName: "test-agent-runtime",
});
const agentRuntimeArtifact = agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, "v1.0.0");

// Optional: Create custom claims for additional validation
const customClaims = [
  agentcore.RuntimeCustomClaim.withStringValue('department', 'engineering'),
  agentcore.RuntimeCustomClaim.withStringArrayValue('roles', ['admin'], agentcore.CustomClaimOperator.CONTAINS),
  agentcore.RuntimeCustomClaim.withStringArrayValue('permissions', ['read', 'write'], agentcore.CustomClaimOperator.CONTAINS_ANY),
];

const runtime = new agentcore.Runtime(this, "MyAgentRuntime", {
  runtimeName: "myAgent",
  agentRuntimeArtifact: agentRuntimeArtifact,
  authorizerConfiguration: agentcore.RuntimeAuthorizerConfiguration.usingCognito(
    userPool, // User Pool (required)
    [userPoolClient, anotherUserPoolClient], // User Pool Clients
    ["audience1"], // Allowed Audiences (optional)
    ["read", "write"], // Allowed Scopes (optional)
    customClaims, // Custom claims (optional) - see Custom Claims Validation section
  ),
});
```

You can configure:

- User Pool: The Cognito User Pool that issues JWT tokens
- User Pool Clients: One or more Cognito User Pool App Clients that are allowed to access the runtime
- Allowed audiences: Used to validate that the audiences specified in the Cognito token match or are a subset of the audiences specified in the AgentCore Runtime
- Allowed scopes: Allow access only if the token contains at least one of the required scopes configured here
- Custom claims: A set of rules to match specific claims in the incoming token against predefined values for validating JWT tokens

#### JWT Authentication

To configure custom JWT authentication with your own OpenID Connect (OIDC) provider:

```typescript fixture=default
const repository = new ecr.Repository(this, "TestRepository", {
  repositoryName: "test-agent-runtime",
});
const agentRuntimeArtifact = agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, "v1.0.0");

const runtime = new agentcore.Runtime(this, "MyAgentRuntime", {
  runtimeName: "myAgent",
  agentRuntimeArtifact: agentRuntimeArtifact,
  authorizerConfiguration: agentcore.RuntimeAuthorizerConfiguration.usingJWT(
    "https://example.com/.well-known/openid-configuration",  // Discovery URL (required)
    ["client1", "client2"],  // Allowed Client IDs (optional)
    ["audience1"],           // Allowed Audiences (optional)
    ["read", "write"],       // Allowed Scopes (optional)
    // Custom claims (optional) - see Custom Claims Validation section below
  ),
});
```

You can configure:

- Discovery URL: Enter the Discovery URL from your identity provider (e.g. Okta, Cognito, etc.), typically found in that provider's documentation. This allows your Agent or Tool to fetch login, downstream resource token, and verification settings.
- Allowed audiences: This is used to validate that the audiences specified for the OAuth token matches or are a subset of the audiences specified in the AgentCore Runtime.
- Allowed clients: This is used to validate that the public identifier of the client, as specified in the authorization token, is allowed to access the AgentCore Runtime.
- Allowed scopes: Allow access only if the token contains at least one of the required scopes configured here.
- Custom claims: A set of rules to match specific claims in the incoming token against predefined values for validating JWT tokens.

**Note**: The discovery URL must end with `/.well-known/openid-configuration`.

##### Custom Claims Validation

Custom claims allow you to validate additional fields in JWT tokens beyond the standard audience, client, and scope validations. You can create custom claims using the `RuntimeCustomClaim` class:

```typescript fixture=default
const repository = new ecr.Repository(this, "TestRepository", {
  repositoryName: "test-agent-runtime",
});
const agentRuntimeArtifact = agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, "v1.0.0");

// String claim - validates that the claim exactly equals the specified value
// Uses EQUALS operator automatically
const departmentClaim = agentcore.RuntimeCustomClaim.withStringValue('department', 'engineering');

// String array claim with CONTAINS operator (default)
// Validates that the claim array contains a specific string value
// IMPORTANT: CONTAINS requires exactly one value in the array parameter
const rolesClaim = agentcore.RuntimeCustomClaim.withStringArrayValue('roles', ['admin']);

// String array claim with CONTAINS_ANY operator
// Validates that the claim array contains at least one of the specified values
// Use this when you want to check for multiple possible values
const permissionsClaim = agentcore.RuntimeCustomClaim.withStringArrayValue(
  'permissions',
  ['read', 'write'],
  agentcore.CustomClaimOperator.CONTAINS_ANY
);

// Use custom claims in authorizer configuration
const runtime = new agentcore.Runtime(this, "MyAgentRuntime", {
  runtimeName: "myAgent",
  agentRuntimeArtifact: agentRuntimeArtifact,
  authorizerConfiguration: agentcore.RuntimeAuthorizerConfiguration.usingJWT(
    "https://example.com/.well-known/openid-configuration",
    ["client1", "client2"],
    ["audience1"],
    ["read", "write"],
    [departmentClaim, rolesClaim, permissionsClaim] // Custom claims
  ),
});
```

**Custom Claim Rules**:

- **String claims**: Must use the `EQUALS` operator (automatically set). The claim value must exactly match the specified string.
- **String array claims**: Can use `CONTAINS` (default) or `CONTAINS_ANY` operators:
  - **`CONTAINS`**: Checks if the claim array contains a specific string value. **Requires exactly one value** in the array parameter. For example, `['admin']` will check if the token's claim array contains the string `'admin'`.
  - **`CONTAINS_ANY`**: Checks if the claim array contains at least one of the provided string values. Use this when you want to validate against multiple possible values. For example, `['read', 'write']` will check if the token's claim array contains either `'read'` or `'write'`.

**Example Use Cases**:

- Use `CONTAINS` when you need to verify a user has a specific role: `RuntimeCustomClaim.withStringArrayValue('roles', ['admin'])`
- Use `CONTAINS_ANY` when you need to verify a user has any of several permissions: `RuntimeCustomClaim.withStringArrayValue('permissions', ['read', 'write'], CustomClaimOperator.CONTAINS_ANY)`

#### OAuth Authentication

To configure OAuth 2.0 authentication:

```typescript fixture=default
const repository = new ecr.Repository(this, "TestRepository", {
  repositoryName: "test-agent-runtime",
});
const agentRuntimeArtifact = agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, "v1.0.0");

const runtime = new agentcore.Runtime(this, "MyAgentRuntime", {
  runtimeName: "myAgent",
  agentRuntimeArtifact: agentRuntimeArtifact,
  authorizerConfiguration: agentcore.RuntimeAuthorizerConfiguration.usingOAuth(
    "https://github.com/.well-known/openid-configuration",  // Discovery URL (required)
    "oauth_client_123",  // OAuth Client ID (required)
    ["audience1"],       // Allowed Audiences (optional)
    ["openid", "profile"], // Allowed Scopes (optional)
    // Custom claims (optional) - see Custom Claims Validation section
  ),
});
```

#### Using a Custom IAM Role

Instead of using the auto-created execution role, you can provide your own IAM role with specific permissions:
The auto-created role includes all necessary baseline permissions for ECR access, CloudWatch logging, and X-Ray tracing. When providing a custom role, ensure these permissions are included.

### Runtime Network Configuration

The AgentCore Runtime supports two network modes for deployment:

#### Public Network Mode (Default)

By default, runtimes are deployed in PUBLIC network mode, which provides internet access suitable for less sensitive or open-use scenarios:

```typescript fixture=default
const repository = new ecr.Repository(this, "TestRepository", {
  repositoryName: "test-agent-runtime",
});
const agentRuntimeArtifact = agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, "v1.0.0");

// Explicitly using public network (this is the default)
const runtime = new agentcore.Runtime(this, "MyAgentRuntime", {
  runtimeName: "myAgent",
  agentRuntimeArtifact: agentRuntimeArtifact,
  networkConfiguration: agentcore.RuntimeNetworkConfiguration.usingPublicNetwork(),
});
```

#### VPC Network Mode

For enhanced security and network isolation, you can deploy your runtime within a VPC:

```typescript fixture=default
const repository = new ecr.Repository(this, "TestRepository", {
  repositoryName: "test-agent-runtime",
});
const agentRuntimeArtifact = agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, "v1.0.0");

// Create or use an existing VPC
const vpc = new ec2.Vpc(this, 'MyVpc', {
  maxAzs: 2,
});

// Configure runtime with VPC
const runtime = new agentcore.Runtime(this, "MyAgentRuntime", {
  runtimeName: "myAgent",
  agentRuntimeArtifact: agentRuntimeArtifact,
  networkConfiguration: agentcore.RuntimeNetworkConfiguration.usingVpc(this, {
    vpc: vpc,
    vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
    // Optionally specify security groups, or one will be created automatically
    // securityGroups: [mySecurityGroup],
  }),
});

```

#### Managing Security Groups with VPC Configuration

When using VPC mode, the Runtime implements `ec2.IConnectable`, allowing you to manage network access using the `connections` property:

```typescript fixture=default
const vpc = new ec2.Vpc(this, 'MyVpc', {
  maxAzs: 2,
});

const repository = new ecr.Repository(this, "TestRepository", {
  repositoryName: "test-agent-runtime",
});
const agentRuntimeArtifact = agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, "v1.0.0");

// Create runtime with VPC configuration
const runtime = new agentcore.Runtime(this, "MyAgentRuntime", {
  runtimeName: "myAgent",
  agentRuntimeArtifact: agentRuntimeArtifact,
  networkConfiguration: agentcore.RuntimeNetworkConfiguration.usingVpc(this, {
    vpc: vpc,
    vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
  }),
});

// Now you can manage network access using the connections property
// Allow inbound HTTPS traffic from a specific security group
const webServerSecurityGroup = new ec2.SecurityGroup(this, 'WebServerSG', { vpc });
runtime.connections.allowFrom(webServerSecurityGroup, ec2.Port.tcp(443), 'Allow HTTPS from web servers');

// Allow outbound connections to a database
const databaseSecurityGroup = new ec2.SecurityGroup(this, 'DatabaseSG', { vpc });
runtime.connections.allowTo(databaseSecurityGroup, ec2.Port.tcp(5432), 'Allow PostgreSQL connection');

// Allow outbound HTTPS to anywhere (for external API calls)
runtime.connections.allowToAnyIpv4(ec2.Port.tcp(443), 'Allow HTTPS outbound');
```

### Runtime IAM Permissions

The Runtime construct provides convenient methods for granting IAM permissions to principals that need to invoke the runtime or manage its execution role.

```typescript fixture=default
const repository = new ecr.Repository(this, "TestRepository", {
  repositoryName: "test-agent-runtime",
});
const agentRuntimeArtifact = agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, "v1.0.0");

// Create a runtime
const runtime = new agentcore.Runtime(this, "MyRuntime", {
  runtimeName: "my_runtime",
  agentRuntimeArtifact: agentRuntimeArtifact,
});

// Create a Lambda function that needs to invoke the runtime
const invokerFunction = new lambda.Function(this, "InvokerFunction", {
  runtime: lambda.Runtime.PYTHON_3_12,
  handler: "index.handler",
  code: lambda.Code.fromInline(`
import boto3
def handler(event, context):
    client = boto3.client('bedrock-agentcore')
    # Invoke the runtime...
  `),
});

// Grant permission to invoke the runtime directly
runtime.grantInvokeRuntime(invokerFunction);

// Grant permission to invoke the runtime on behalf of a user
// (requires X-Amzn-Bedrock-AgentCore-Runtime-User-Id header)
runtime.grantInvokeRuntimeForUser(invokerFunction);

// Grant both invoke permissions (most common use case)
runtime.grantInvoke(invokerFunction);

// Grant specific custom permissions to the runtime's execution role
runtime.grant(['bedrock:InvokeModel'], ['arn:aws:bedrock:*:*:*']);

// Add a policy statement to the runtime's execution role
runtime.addToRolePolicy(new iam.PolicyStatement({
  actions: ['s3:GetObject'],
  resources: ['arn:aws:s3:::my-bucket/*'],
}));
```

### Other configuration

#### Lifecycle configuration

The LifecycleConfiguration input parameter to CreateAgentRuntime lets you manage the lifecycle of runtime sessions and resources in Amazon Bedrock AgentCore Runtime. This configuration helps optimize resource utilization by automatically cleaning up idle sessions and preventing long-running instances from consuming resources indefinitely.

You can configure:

- idleRuntimeSessionTimeout: Timeout in seconds for idle runtime sessions. When a session remains idle for this duration, it will trigger termination. Termination can last up to 15 seconds due to logging and other process completion. Default: 900 seconds (15 minutes)
- maxLifetime: Maximum lifetime for the instance in seconds. Once reached, instances will initialize termination. Termination can last up to 15 seconds due to logging and other process completion. Default: 28800 seconds (8 hours)

For additional information, please refer to the [documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-lifecycle-settings.html).

```typescript fixture=default
const repository = new ecr.Repository(this, "TestRepository", {
  repositoryName: "test-agent-runtime",
});

const agentRuntimeArtifact = agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, "v1.0.0");

new agentcore.Runtime(this, 'test-runtime', {
  runtimeName: 'test_runtime',
  agentRuntimeArtifact: agentRuntimeArtifact,
  lifecycleConfiguration: {
    idleRuntimeSessionTimeout: Duration.minutes(10),
    maxLifetime: Duration.hours(4),
  },
});
```

#### Request header configuration

Custom headers let you pass contextual information from your application directly to your agent code without cluttering the main request payload. This includes authentication tokens like JWT (JSON Web Tokens, which contain user identity and authorization claims) through the Authorization header, allowing your agent to make decisions based on who is calling it. You can also pass custom metadata like user preferences, session identifiers, or trace context using headers prefixed with X-Amzn-Bedrock-AgentCore-Runtime-Custom-, giving your agent access to up to 20 pieces of runtime context that travel alongside each request. This information can be also used in downstream systems like AgentCore Memory that you can namespace based on those characteristics like user_id or aud in claims like line of business.

For additional information, please refer to the [documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-header-allowlist.html).

```typescript fixture=default
const repository = new ecr.Repository(this, "TestRepository", {
  repositoryName: "test-agent-runtime",
});

const agentRuntimeArtifact = agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, "v1.0.0");

new agentcore.Runtime(this, 'test-runtime', {
  runtimeName: 'test_runtime',
  agentRuntimeArtifact: agentRuntimeArtifact,
  requestHeaderConfiguration: {
    allowlistedHeaders: ['X-Amzn-Bedrock-AgentCore-Runtime-Custom-H1'],
  },
});
```

#### Observability configuration

The Runtime construct supports observability features including X-Ray tracing and logging to CloudWatch Logs, S3, or Kinesis Data Firehose. This allows you to monitor and debug your agent runtime invocations.

You can configure:

- tracingEnabled: Enable X-Ray tracing for the runtime
- loggingConfigs: Send APPLICATION_LOGS (agent runtime invocations) and USAGE_LOGS (session-level resource consumption) to CloudWatch Logs, S3, or Kinesis Data Firehose

For additional information, please refer to the [Set up logging and tracing for AgentCore](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html).

```typescript fixture=default
const repository = new ecr.Repository(this, 'TestRepository', {
  repositoryName: 'test-agent-runtime',
});

const agentRuntimeArtifact = agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');

// Create logging destinations
const logGroup = new logs.LogGroup(this, 'RuntimeLogGroup');
const logBucket = new s3.Bucket(this, 'RuntimeLogBucket');
const firehoseStream = new firehose.DeliveryStream(this, 'RuntimeLogStream', {
  destination: new firehose.S3Bucket(logBucket),
});

new agentcore.Runtime(this, 'test-runtime', {
  runtimeName: 'test_runtime',
  agentRuntimeArtifact: agentRuntimeArtifact,
  tracingEnabled: true,
  loggingConfigs: [
    {
      logType: agentcore.LogType.APPLICATION_LOGS,
      destination: agentcore.LoggingDestination.cloudWatchLogs(logGroup),
    },
    {
      logType: agentcore.LogType.APPLICATION_LOGS,
      destination: agentcore.LoggingDestination.s3(logBucket),
    },
    {
      logType: agentcore.LogType.APPLICATION_LOGS,
      destination: agentcore.LoggingDestination.firehose(firehoseStream),
    },
  ],
});
```

#### Application log group

Every Runtime has a default endpoint whose stdout is written to the AgentCore-managed log group at `/aws/bedrock-agentcore/runtimes/{agentRuntimeId}-DEFAULT`. The Runtime construct exposes this log group as `applicationLogGroup` so you can attach metric filters, subscription filters, or alarms without hardcoding the path:

```typescript fixture=default
const repository = new ecr.Repository(this, 'TestRepository');

const runtime = new agentcore.Runtime(this, 'Runtime', {
  agentRuntimeArtifact: agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0'),
});

new logs.MetricFilter(this, 'ToolErrors', {
  logGroup: runtime.applicationLogGroup,
  filterPattern: logs.FilterPattern.stringValue('$.tool_status', '=', 'error'),
  metricNamespace: 'MyApp',
  metricName: 'ToolExecutionErrors',
});
```

The log group itself is created by the AgentCore service on the runtime's first invocation, not by CDK. Constructs that require the log group to exist at deploy time may race the first invocation; if that is a concern, pre-create the log group with a `LogRetention` resource using the same name.

## Browser

The Amazon Bedrock AgentCore Browser provides a secure, cloud-based browser that enables AI agents to interact with websites. It includes security features such as session isolation, built-in observability through live viewing, CloudTrail logging, and session replay capabilities.

Additional information about the browser tool can be found in the [official documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/browser-tool.html)

### Browser Network modes

The Browser construct supports the following network modes:

1. **Public Network Mode** (`BrowserNetworkMode.usingPublicNetwork()`) - Default

   - Allows internet access for web browsing and external API calls
   - Suitable for scenarios where agents need to interact with publicly available websites
   - Enables full web browsing capabilities
   - VPC mode is not supported with this option

2. **VPC (Virtual Private Cloud)** (`BrowserNetworkMode.usingVpc()`)

   - Select whether to run the browser in a virtual private cloud (VPC).
   - By configuring VPC connectivity, you enable secure access to private resources such as databases, internal APIs, and services within your VPC.

    While the VPC itself is mandatory, these are optional:
    - Subnets - if not provided, CDK will select appropriate subnets from the VPC
    - Security Groups - if not provided, CDK will create a default security group
    - Specific subnet selection criteria - you can let CDK choose automatically

For more information on VPC connectivity for Amazon Bedrock AgentCore Browser, please refer to the [official documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-vpc.html).

### Browser Properties

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `browserCustomName` | `string` | No | The name of the browser. Must start with a letter and can be up to 48 characters long. Pattern: `[a-zA-Z][a-zA-Z0-9_]{0,47}`. If not provided, a unique name will be auto-generated |
| `description` | `string` | No | Optional description for the browser. Can have up to 200 characters |
| `networkConfiguration` | `BrowserNetworkConfiguration` | No | Network configuration for browser. Defaults to PUBLIC network mode |
| `recordingConfig` | `RecordingConfig` | No | Recording configuration for browser. Defaults to no recording |
| `executionRole` | `iam.IRole` | No | The IAM role that provides permissions for the browser to access AWS services. A new role will be created if not provided |
| `tags` | `{ [key: string]: string }` | No | Tags to apply to the browser resource |
| `browserSigning` | BrowserSigning | No | Browser signing configuration. Defaults to DISABLED |

### Basic Browser Creation

```typescript fixture=default
// Create a basic browser with public network access
const browser = new agentcore.BrowserCustom(this, "MyBrowser", {
  browserCustomName: "my_browser",
  description: "A browser for web automation",
});
```

### Browser with Tags

```typescript fixture=default
// Create a browser with custom tags
const browser = new agentcore.BrowserCustom(this, "MyBrowser", {
  browserCustomName: "my_browser",
  description: "A browser for web automation with tags",
  networkConfiguration: agentcore.BrowserNetworkConfiguration.usingPublicNetwork(),
  tags: {
    Environment: "Production",
    Team: "AI/ML",
    Project: "AgentCore",
  },
});
```

### Browser with VPC

```typescript fixture=default
const browser = new agentcore.BrowserCustom(this, 'BrowserVpcWithRecording', {
  browserCustomName: 'browser_recording',
  networkConfiguration: agentcore.BrowserNetworkConfiguration.usingVpc(this, {
    vpc: new ec2.Vpc(this, 'VPC', { restrictDefaultSecurityGroup: false }),
  }),
});
```

Browser exposes a [connections](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_ec2.Connections.html) property. This property returns a connections object, which simplifies the process of defining and managing ingress and egress rules for security groups in your AWS CDK applications. Instead of directly manipulating security group rules, you interact with the Connections object of a construct, which then translates your connectivity requirements into the appropriate security group rules. For instance:

```typescript fixture=default
const vpc = new ec2.Vpc(this, 'testVPC');

const browser = new agentcore.BrowserCustom(this, 'test-browser', {
  browserCustomName: 'test_browser',
  networkConfiguration: agentcore.BrowserNetworkConfiguration.usingVpc(this, {
    vpc: vpc,
  }),
});

browser.connections.addSecurityGroup(new ec2.SecurityGroup(this, 'AdditionalGroup', { vpc }));
```

So security groups can be added after the browser construct creation. You can use methods like allowFrom() and allowTo() to grant ingress access to/egress access from a specified peer over a given portRange. The Connections object automatically adds the necessary ingress or egress rules to the security group(s) associated with the calling construct.

### Browser with Recording Configuration

```typescript fixture=default
// Create an S3 bucket for recordings
const recordingBucket = new s3.Bucket(this, "RecordingBucket", {
  bucketName: "my-browser-recordings",
  removalPolicy: RemovalPolicy.DESTROY, // For demo purposes
});

// Create browser with recording enabled
const browser = new agentcore.BrowserCustom(this, "MyBrowser", {
  browserCustomName: "my_browser",
  description: "Browser with recording enabled",
  networkConfiguration: agentcore.BrowserNetworkConfiguration.usingPublicNetwork(),
  recordingConfig: {
    enabled: true,
    s3Location: {
      bucketName: recordingBucket.bucketName,
      objectKey: "browser-recordings/",
    },
  },
});
```

### Browser with Custom Execution Role

```typescript fixture=default
// Create a custom execution role
const executionRole = new iam.Role(this, "BrowserExecutionRole", {
  assumedBy: new iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
  managedPolicies: [
    iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonBedrockAgentCoreBrowserExecutionRolePolicy"),
  ],
});

// Create browser with custom execution role
const browser = new agentcore.BrowserCustom(this, "MyBrowser", {
  browserCustomName: "my_browser",
  description: "Browser with custom execution role",
  networkConfiguration: agentcore.BrowserNetworkConfiguration.usingPublicNetwork(),
  executionRole: executionRole,
});
```

### Browser with S3 Recording and Permissions

```typescript fixture=default
// Create an S3 bucket for recordings
const recordingBucket = new s3.Bucket(this, "RecordingBucket", {
  bucketName: "my-browser-recordings",
  removalPolicy: RemovalPolicy.DESTROY, // For demo purposes
});

// Create browser with recording enabled
const browser = new agentcore.BrowserCustom(this, "MyBrowser", {
  browserCustomName: "my_browser",
  description: "Browser with recording enabled",
  networkConfiguration: agentcore.BrowserNetworkConfiguration.usingPublicNetwork(),
  recordingConfig: {
    enabled: true,
    s3Location: {
      bucketName: recordingBucket.bucketName,
      objectKey: "browser-recordings/",
    },
  },
});

// The browser construct automatically grants S3 permissions to the execution role
// when recording is enabled, so no additional IAM configuration is needed
```

### Browser with Browser signing

AI agents need to browse the web on your behalf. When your agent visits a website to gather information, complete a form, or verify data, it encounters the same defenses designed to stop unwanted bots: CAPTCHAs, rate limits, and outright blocks.

Amazon Bedrock AgentCore Browser supports Web Bot Auth. Web Bot Auth is a draft IETF protocol that gives agents verifiable cryptographic identities. When you enable Web Bot Auth in AgentCore Browser, the service issues cryptographic credentials that websites can verify. The agent presents these credentials with every request. The WAF may now additionally check the signature, confirm it matches a trusted directory, and allow the request through if verified bots are allowed by the domain owner and other WAF checks are clear.

To enable the browser to sign requests using the Web Bot Auth protocol, create a browser tool with the browserSigning configuration:

```typescript fixture=default
const browser = new agentcore.BrowserCustom(this, 'test-browser', {
  browserCustomName: 'test_browser',
  browserSigning: agentcore.BrowserSigning.ENABLED
});
```

### Browser IAM Permissions

The Browser construct provides convenient methods for granting IAM permissions:

```typescript fixture=default
// Create a browser
const browser = new agentcore.BrowserCustom(this, "MyBrowser", {
  browserCustomName: "my_browser",
  description: "Browser for web automation",
  networkConfiguration: agentcore.BrowserNetworkConfiguration.usingPublicNetwork(),
});

// Create a role that needs access to the browser
const userRole = new iam.Role(this, "UserRole", {
  assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
});

// Grant read permissions (Get and List actions)
browser.grantRead(userRole);

// Grant use permissions (Start, Update, Stop actions)
browser.grantUse(userRole);

// Grant specific custom permissions
browser.grant(userRole, "bedrock-agentcore:GetBrowserSession");
```

## Code Interpreter

The Amazon Bedrock AgentCore Code Interpreter enables AI agents to write and execute code securely in sandbox environments, enhancing their accuracy and expanding their ability to solve complex end-to-end tasks. This is critical in Agentic AI applications where the agents may execute arbitrary code that can lead to data compromise or security risks. The AgentCore Code Interpreter tool provides secure code execution, which helps you avoid running into these issues.

For more information about code interpreter, please refer to the [official documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html)

### Code Interpreter Network Modes

The Code Interpreter construct supports the following network modes:

1. **Public Network Mode** (`CodeInterpreterNetworkMode.usingPublicNetwork()`) - Default

   - Allows internet access for package installation and external API calls
   - Suitable for development and testing environments
   - Enables downloading Python packages from PyPI

2. **Sandbox Network Mode** (`CodeInterpreterNetworkMode.usingSandboxNetwork()`)
   - Isolated network environment with no internet access
   - Suitable for production environments with strict security requirements
   - Only allows access to pre-installed packages and local resources

3. **VPC (Virtual Private Cloud)** (`CodeInterpreterNetworkMode.usingVpc()`)
   - Select whether to run the browser in a virtual private cloud (VPC).
   - By configuring VPC connectivity, you enable secure access to private resources such as databases, internal APIs, and services within your VPC.

    While the VPC itself is mandatory, these are optional:
    - Subnets - if not provided, CDK will select appropriate subnets from the VPC
    - Security Groups - if not provided, CDK will create a default security group
    - Specific subnet selection criteria - you can let CDK choose automatically

For more information on VPC connectivity for Amazon Bedrock AgentCore Browser, please refer to the [official documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-vpc.html).

### Code Interpreter Properties

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `codeInterpreterCustomName` | `string` | No | The name of the code interpreter. Must start with a letter and can be up to 48 characters long. Pattern: `[a-zA-Z][a-zA-Z0-9_]{0,47}`. If not provided, a unique name will be auto-generated |
| `description` | `string` | No | Optional description for the code interpreter. Can have up to 200 characters |
| `executionRole` | `iam.IRole` | No | The IAM role that provides permissions for the code interpreter to access AWS services. A new role will be created if not provided |
| `networkConfiguration` | `CodeInterpreterNetworkConfiguration` | No | Network configuration for code interpreter. Defaults to PUBLIC network mode |
| `tags` | `{ [key: string]: string }` | No | Tags to apply to the code interpreter resource |

### Basic Code Interpreter Creation

```typescript fixture=default
// Create a basic code interpreter with public network access
const codeInterpreter = new agentcore.CodeInterpreterCustom(this, "MyCodeInterpreter", {
  codeInterpreterCustomName: "my_code_interpreter",
  description: "A code interpreter for Python execution",
});
```

### Code Interpreter with VPC

```typescript fixture=default
const codeInterpreter = new agentcore.CodeInterpreterCustom(this, "MyCodeInterpreter", {
  codeInterpreterCustomName: "my_sandbox_interpreter",
  description: "Code interpreter with isolated network access",
  networkConfiguration: agentcore.BrowserNetworkConfiguration.usingVpc(this, {
    vpc: new ec2.Vpc(this, 'VPC', { restrictDefaultSecurityGroup: false }),
  }),
});
```

Code Interpreter exposes a [connections](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_ec2.Connections.html) property. This property returns a connections object, which simplifies the process of defining and managing ingress and egress rules for security groups in your AWS CDK applications. Instead of directly manipulating security group rules, you interact with the Connections object of a construct, which then translates your connectivity requirements into the appropriate security group rules. For instance:

```typescript fixture=default
const vpc = new ec2.Vpc(this, 'testVPC');

const codeInterpreter = new agentcore.CodeInterpreterCustom(this, "MyCodeInterpreter", {
  codeInterpreterCustomName: "my_sandbox_interpreter",
  description: "Code interpreter with isolated network access",
  networkConfiguration: agentcore.BrowserNetworkConfiguration.usingVpc(this, {
    vpc: vpc,
  }),
});

codeInterpreter.connections.addSecurityGroup(new ec2.SecurityGroup(this, 'AdditionalGroup', { vpc }));
```

So security groups can be added after the browser construct creation. You can use methods like allowFrom() and allowTo() to grant ingress access to/egress access from a specified peer over a given portRange. The Connections object automatically adds the necessary ingress or egress rules to the security group(s) associated with the calling construct.

### Code Interpreter with Sandbox Network Mode

```typescript fixture=default
// Create code interpreter with sandbox network mode (isolated)
const codeInterpreter = new agentcore.CodeInterpreterCustom(this, "MyCodeInterpreter", {
  codeInterpreterCustomName: "my_sandbox_interpreter",
  description: "Code interpreter with isolated network access",
  networkConfiguration: agentcore.CodeInterpreterNetworkConfiguration.usingSandboxNetwork(),
});
```

### Code Interpreter with Custom Execution Role

```typescript fixture=default
// Create a custom execution role
const executionRole = new iam.Role(this, "CodeInterpreterExecutionRole", {
  assumedBy: new iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
});

// Create code interpreter with custom execution role
const codeInterpreter = new agentcore.CodeInterpreterCustom(this, "MyCodeInterpreter", {
  codeInterpreterCustomName: "my_code_interpreter",
  description: "Code interpreter with custom execution role",
  networkConfiguration: agentcore.CodeInterpreterNetworkConfiguration.usingPublicNetwork(),
  executionRole: executionRole,
});
```

### Code Interpreter IAM Permissions

The Code Interpreter construct provides convenient methods for granting IAM permissions:

```typescript fixture=default
// Create a code interpreter
const codeInterpreter = new agentcore.CodeInterpreterCustom(this, "MyCodeInterpreter", {
  codeInterpreterCustomName: "my_code_interpreter",
  description: "Code interpreter for Python execution",
  networkConfiguration: agentcore.CodeInterpreterNetworkConfiguration.usingPublicNetwork(),
});

// Create a role that needs access to the code interpreter
const userRole = new iam.Role(this, "UserRole", {
  assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
});

// Grant read permissions (Get and List actions)
codeInterpreter.grantRead(userRole);

// Grant use permissions (Start, Invoke, Stop actions)
codeInterpreter.grantUse(userRole);

// Grant specific custom permissions
codeInterpreter.grant(userRole, "bedrock-agentcore:GetCodeInterpreterSession");
```

### Code interpreter with tags

```typescript fixture=default
// Create code interpreter with sandbox network mode (isolated)
const codeInterpreter = new agentcore.CodeInterpreterCustom(this, "MyCodeInterpreter", {
  codeInterpreterCustomName: "my_sandbox_interpreter",
  description: "Code interpreter with isolated network access",
  networkConfiguration: agentcore.CodeInterpreterNetworkConfiguration.usingPublicNetwork(),
  tags: {
    Environment: "Production",
    Team: "AI/ML",
    Project: "AgentCore",
  },
});
```

## Gateway

The Gateway construct provides a way to create Amazon Bedrock Agent Core Gateways, which serve as integration points between agents and external services.

### Gateway Properties

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `gatewayName` | `string` | No | The name of the gateway. Valid characters are a-z, A-Z, 0-9, _ (underscore) and - (hyphen). Maximum 100 characters. If not provided, a unique name will be auto-generated |
| `description` | `string` | No | Optional description for the gateway. Maximum 200 characters |
| `protocolConfiguration` | `IGatewayProtocolConfig` | No | The protocol configuration for the gateway. Defaults to MCP protocol |
| `authorizerConfiguration` | `IGatewayAuthorizerConfig` | No | The authorizer configuration for the gateway. Defaults to Cognito |
| `exceptionLevel` | `GatewayExceptionLevel` | No | The verbosity of exception messages. Use DEBUG mode to see granular exception messages |
| `kmsKey` | `kms.IKey` | No | The AWS KMS key used to encrypt data associated with the gateway |
| `role` | `iam.IRole` | No | The IAM role that provides permissions for the gateway to access AWS services. A new role will be created if not provided |
| `tags` | `{ [key: string]: string }` | No | Tags for the gateway. A list of key:value pairs of tags to apply to this Gateway resource |

### Basic Gateway Creation

The protocol configuration defaults to MCP and the inbound auth configuration uses Cognito (it is automatically created on your behalf).

```typescript fixture=default
// Create a basic gateway with default MCP protocol and Cognito authorizer
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
});
```

### Protocol configuration

Currently MCP is the only protocol available. To configure it, use the `protocol` property with `McpProtocolConfiguration`:

- Instructions: Guidance for how to use the gateway with your tools
- Semantic search: Smart tool discovery that finds the right tools without typical limits. It improves accuracy by finding relevant tools based on context
- Supported versions: Which MCP protocol versions the gateway can use

```typescript fixture=default
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
  protocolConfiguration: new agentcore.McpProtocolConfiguration({
    instructions: "Use this gateway to connect to external MCP tools",
    searchType: agentcore.McpGatewaySearchType.SEMANTIC,
    supportedVersions: [agentcore.MCPProtocolVersion.MCP_2025_03_26],
  }),
});
```

### Inbound authorization

Before you create your gateway, you must set up inbound authorization. Inbound authorization validates users who attempt to access targets through
your AgentCore gateway. By default, if not provided, the construct will create and configure Cognito as the default identity provider
(inbound Auth setup). AgentCore supports the following types of inbound authorization:

**JSON Web Token (JWT)** – A secure and compact token used for authorization. After creating the JWT, you specify it as the authorization
configuration when you create the gateway. You can create a JWT with any of the identity providers at Provider setup and configuration.

You can configure a custom authorization provider using the `authorizerConfiguration` property with `GatewayAuthorizer.usingCustomJwt()`.
You need to specify an OAuth discovery server and client IDs/audiences when you create the gateway. You can specify the following:

- Discovery Url — String that must match the pattern ^.+/\.well-known/openid-configuration$ for OpenID Connect discovery URLs
- At least one of the below options depending on the chosen identity provider.
- Allowed audiences — List of allowed audiences for JWT tokens
- Allowed clients — List of allowed client identifiers
- Allowed scopes — List of allowed scopes for JWT tokens
- Custom claims — Optional custom claim validations (see Custom Claims Validation section below)

```typescript fixture=default

// Optional: Create custom claims (CustomClaimOperator and GatewayCustomClaim from agentcore)
const customClaims = [
  agentcore.GatewayCustomClaim.withStringValue('department', 'engineering'),
  agentcore.GatewayCustomClaim.withStringArrayValue('roles', ['admin'], agentcore.CustomClaimOperator.CONTAINS),
  agentcore.GatewayCustomClaim.withStringArrayValue('permissions', ['read', 'write'], agentcore.CustomClaimOperator.CONTAINS_ANY),
];

const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
  authorizerConfiguration: agentcore.GatewayAuthorizer.usingCustomJwt({
    discoveryUrl: "https://auth.example.com/.well-known/openid-configuration",
    allowedAudience: ["my-app"],
    allowedClients: ["my-client-id"],
    allowedScopes: ["read", "write"],
    customClaims: customClaims, // Optional custom claims
  }),
});
```

**IAM** – Authorizes through the credentials of the AWS IAM identity trying to access the gateway.

```typescript fixture=default
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
  authorizerConfiguration: agentcore.GatewayAuthorizer.usingAwsIam(),
});

// Grant access to a Lambda function's role
const lambdaRole = new iam.Role(this, "LambdaRole", {
  assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
});

// The Lambda needs permission to invoke the gateway
gateway.grantInvoke(lambdaRole);
```

**No Authorization** – Creates a gateway with no inbound authorization. This is useful for building public MCP servers,
or when you want to skip gateway-level authentication and enforce tool execution-level authentication using Gateway Interceptors.

```typescript fixture=default
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
  authorizerConfiguration: agentcore.GatewayAuthorizer.withNoAuth(),
});
```

> **⚠️ Important:** Do not use No Authorization gateways for production workloads unless you have implemented all the security best practices. No Authorization gateways are most appropriate for testing and development purposes. See [Security Best Practices](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-inbound-auth.html#gateway-inbound-auth-none) for required compensating controls.

For more information, see [No Authorization](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-inbound-auth.html#gateway-inbound-auth-none).

**Cognito with M2M (Machine-to-Machine) Authentication (Default)** – When no authorizer is specified, the construct automatically creates a Cognito User Pool configured for OAuth 2.0 client credentials flow. This enables machine-to-machine authentication suitable for AI agents and service-to-service communication.

For more information, see [Setting up Amazon Cognito for Gateway inbound authorization](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/identity-idp-cognito.html).

```typescript fixture=default
// Create a gateway with default Cognito M2M authorizer
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
});

// Access the Cognito resources for authentication setup
const userPool = gateway.userPool;
const userPoolClient = gateway.userPoolClient;

// Get the token endpoint URL and OAuth scopes for client credentials flow
const tokenEndpointUrl = gateway.tokenEndpointUrl;
const oauthScopes = gateway.oauthScopes;
// oauthScopes are in the format: ['{resourceServerId}/read', '{resourceServerId}/write']
```

**Using Cognito User Pool Explicitly with Custom Claims** – You can also use an existing Cognito User Pool with custom claims:

```typescript fixture=default
declare const userPool: cognito.UserPool;
declare const userPoolClient: cognito.UserPoolClient;

// Optional: Create custom claims (CustomClaimOperator and GatewayCustomClaim from agentcore)
const customClaims = [
  agentcore.GatewayCustomClaim.withStringValue('department', 'engineering'),
  agentcore.GatewayCustomClaim.withStringArrayValue('roles', ['admin'], agentcore.CustomClaimOperator.CONTAINS),
  agentcore.GatewayCustomClaim.withStringArrayValue('permissions', ['read', 'write'], agentcore.CustomClaimOperator.CONTAINS_ANY),
];

const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
  authorizerConfiguration: agentcore.GatewayAuthorizer.usingCognito({
    userPool: userPool,
    allowedClients: [userPoolClient],
    allowedAudiences: ["audience1"],
    allowedScopes: ["read", "write"],
    customClaims: customClaims, // Optional custom claims
  }),
});
```

To authenticate with the gateway, request an access token using the client credentials flow and use it to call Gateway endpoints. For more information about the token endpoint, see [The token issuer endpoint](https://docs.aws.amazon.com/cognito/latest/developerguide/token-endpoint.html).

The following is an example of a token request using curl:

```bash
curl -X POST "${TOKEN_ENDPOINT_URL}" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=${USER_POOL_CLIENT_ID}" \
  -d "client_secret=${CLIENT_SECRET}" \
  -d "scope=${OAUTH_SCOPES}"
```

### Gateway with KMS Encryption

You can provide a KMS key, and configure the authorizer as well as the protocol configuration.

```typescript fixture=default
// Create a KMS key for encryption
const encryptionKey = new kms.Key(this, "GatewayEncryptionKey", {
  enableKeyRotation: true,
  description: "KMS key for gateway encryption",
});

// Create gateway with KMS encryption
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-encrypted-gateway",
  description: "Gateway with KMS encryption",
  protocolConfiguration: new agentcore.McpProtocolConfiguration({
    instructions: "Use this gateway to connect to external MCP tools",
    searchType: agentcore.McpGatewaySearchType.SEMANTIC,
    supportedVersions: [agentcore.MCPProtocolVersion.MCP_2025_03_26],
  }),
  authorizerConfiguration: agentcore.GatewayAuthorizer.usingCustomJwt({
    discoveryUrl: "https://auth.example.com/.well-known/openid-configuration",
    allowedAudience: ["my-app"],
    allowedClients: ["my-client-id"],
    allowedScopes: ["read", "write"],
  }),
  kmsKey: encryptionKey,
  exceptionLevel: agentcore.GatewayExceptionLevel.DEBUG,
});
```

### Gateway with Custom Execution Role

```typescript fixture=default
// Create a custom execution role
const executionRole = new iam.Role(this, "GatewayExecutionRole", {
  assumedBy: new iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
  managedPolicies: [
    iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonBedrockAgentCoreGatewayExecutionRolePolicy"),
  ],
});

// Create gateway with custom execution role
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
  description: "Gateway with custom execution role",
  protocolConfiguration: new agentcore.McpProtocolConfiguration({
    instructions: "Use this gateway to connect to external MCP tools",
    searchType: agentcore.McpGatewaySearchType.SEMANTIC,
    supportedVersions: [agentcore.MCPProtocolVersion.MCP_2025_03_26],
  }),
  authorizerConfiguration: agentcore.GatewayAuthorizer.usingCustomJwt({
    discoveryUrl: "https://auth.example.com/.well-known/openid-configuration",
    allowedAudience: ["my-app"],
    allowedClients: ["my-client-id"],
    allowedScopes: ["read", "write"],
  }),
  role: executionRole,
});
```

### Gateway IAM Permissions

The Gateway construct provides convenient methods for granting IAM permissions:

```typescript fixture=default
// Create a gateway
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
  description: "Gateway for external service integration",
  protocolConfiguration: new agentcore.McpProtocolConfiguration({
    instructions: "Use this gateway to connect to external MCP tools",
    searchType: agentcore.McpGatewaySearchType.SEMANTIC,
    supportedVersions: [agentcore.MCPProtocolVersion.MCP_2025_03_26],
  }),
  authorizerConfiguration: agentcore.GatewayAuthorizer.usingCustomJwt({
    discoveryUrl: "https://auth.example.com/.well-known/openid-configuration",
    allowedAudience: ["my-app"],
    allowedClients: ["my-client-id"],
    allowedScopes: ["read", "write"],
  }),
});

// Create a role that needs access to the gateway
const userRole = new iam.Role(this, "UserRole", {
  assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
});

// Grant read permissions (Get and List actions)
gateway.grantRead(userRole);

// Grant manage permissions (Create, Update, Delete actions)
gateway.grantManage(userRole);

// Grant specific custom permissions
gateway.grant(userRole, "bedrock-agentcore:GetGateway");
```

## Gateway Target

After Creating gateways, you can add targets which define the tools that your gateway will host. Gateway supports multiple target
types including Lambda functions and API specifications (either OpenAPI schemas or Smithy models). Gateway allows you to attach multiple
targets to a Gateway and you can change the targets / tools attached to a gateway at any point. Each target can have its own
credential provider attached enabling you to securely access targets whether they need IAM, API Key, or OAuth credentials.

### Gateway Target Properties

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `gatewayTargetName` | `string` | No | The name of the gateway target. Valid characters are a-z, A-Z, 0-9, _ (underscore) and - (hyphen). If not provided, a unique name will be auto-generated |
| `description` | `string` | No | Optional description for the gateway target. Maximum 200 characters |
| `gateway` | `IGateway` | Yes | The gateway this target belongs to |
| `targetConfiguration` | `ITargetConfiguration` | Yes | The target configuration (Lambda, OpenAPI, Smithy, or API Gateway). **Note:** Users typically don't create this directly. When using convenience methods like `GatewayTarget.forLambda()`, `GatewayTarget.forOpenApi()`, `GatewayTarget.forSmithy()`, `GatewayTarget.forApiGateway()`, `GatewayTarget.forMcpServer()` or the gateway's `addLambdaTarget()`, `addOpenApiTarget()`, `addSmithyTarget()`, `addApiGatewayTarget()`, `addMcpServerTarget()` methods, this configuration is created internally for you. Only needed when using the GatewayTarget constructor directly for [advanced scenarios](#advanced-usage-direct-configuration-for-gateway-target). |
| `credentialProviderConfigurations` | `IGatewayCredentialProvider[]` | No | Credential providers for authentication. Defaults to `[GatewayCredentialProvider.fromIamRole()]`. With Token Vault L2 constructs, prefer `GatewayCredentialProvider.fromApiKeyIdentity()` / `fromOauthIdentity()`; otherwise use `fromApiKeyIdentityArn()` / `fromOauthIdentityArn()`, or `fromIamRole()` |
| `validateOpenApiSchema` | `boolean` | No | (OpenAPI targets only) Whether to validate the OpenAPI schema at synthesis time. Defaults to `true`. Only applies to inline and local asset schemas. For more information refer here <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-schema-openapi.html> |

This approach gives you full control over the configuration but is typically not necessary for most use cases.

### Targets types

You can create the following targets types:

**Lambda Target**: Lambda targets allow you to connect your gateway to AWS Lambda functions that implement your tools. This is useful
when you want to execute custom code in response to tool invocations.

- Supports GATEWAY_IAM_ROLE credential provider only
- Ideal for custom serverless function integration
- Need tool schema (tool schema is a blueprint that describes the functions your Lambda provides to AI agents).
  The construct provide [3 ways to upload a tool schema to Lambda target](#tools-schema-for-lambda-target)
- When using the default IAM authentication (no `credentialProviderConfigurations` specified),
  the construct automatocally grants the gateway role permission to invoke your Lambda function (`lambda:InvokeFunction`).

**OpenAPI Schema Target** : OpenAPI widely used standard for describing RESTful APIs. Gateway supports OpenAPI 3.0
specifications for defining API targets. It connects to REST APIs using OpenAPI specifications

- Supports OAUTH and API_KEY credential providers (Do not support IAM, you must provide `credentialProviderConfigurations`)
- Ideal for integrating with external REST services
- Need API schema. The construct provide [3 ways to upload a API schema to OpenAPI target](#api-schema-for-openapi-and-smithy-target)

**Smithy Model Target** : Smithy is a language for defining services and software development kits (SDKs). Smithy models provide
a more structured approach to defining APIs compared to OpenAPI, and are particularly useful for connecting to AWS services.
AgentCore Gateway supports built-in AWS service models only. It connects to services using Smithy model definitions

- Supports OAUTH and API_KEY credential providers
- Ideal for AWS service integrations
- Need API schema. The construct provide 3 ways to upload a API schema to Smity target
- When using the default IAM authentication (no `credentialProviderConfigurations` specified), The construct only
  grants permission to read the Smithy schema file from S3. You MUST manually grant permissions for the gateway
  role to invoke the actual Smithy API endpoints

> Note: For Smithy model targets that access AWS services, your Gateway's execution role needs permissions to access those services.
For example, for a DynamoDB target, your execution role needs permissions to perform DynamoDB operations.
This is not managed by the construct due to the large number of options. Please refer to
[Smithy Model Permission](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-prerequisites-permissions.html) for example.

**MCP Server Target**: Model Context Protocol (MCP) servers provide external tools, data access, and custom functions for AI agents.
MCP servers enable agents to interact with external systems and services through a standardized protocol. Gateway automatically
discovers and indexes available tools from MCP servers through synchronization.

**Key Features:**

- Requires explicit authentication configuration (OAuth2 recommended, empty array for NoAuth)
- Ideal for connecting to external MCP-compliant servers
- The endpoint must use HTTPS protocol
- Supported MCP protocol versions: 2025-06-18, 2025-03-26
- Automatic tool discovery through synchronization

**Synchronization Behavior:**

MCP Server targets require synchronization to discover and index available tools:

- **Implicit Synchronization (Automatic)**: Tool discovery happens automatically during:
  - Target creation (`CreateGatewayTarget`)
  - Target updates (`UpdateGatewayTarget`)
  - The Gateway calls the MCP server's `tools/list` endpoint and indexes tools without user intervention

- **Explicit Synchronization (Manual)**: When the MCP server's tools change independently (new tools added, schemas modified, tools removed):
  - The Gateway's tool catalog becomes stale
  - Call the `SynchronizeGatewayTargets` API to refresh the catalog
  - Use the `grantSync()` method to grant permissions to Lambda functions, CI/CD pipelines, or scheduled tasks that will trigger synchronization

**Authentication & Permissions:**

When using OAuth2, the Gateway service role automatically receives:

- `bedrock-agentcore:GetWorkloadAccessToken`
- `bedrock-agentcore:GetResourceOauth2Token`
- `secretsmanager:GetSecretValue`
- KMS decrypt (if secrets are encrypted)

For explicit synchronization, use `grantSync()` to grant `bedrock-agentcore:SynchronizeGatewayTargets` permission to your operator roles.

> For more information, refer to the [MCP Server Target documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-target-MCPservers.html).

### Understanding Tool Naming

When tools are exposed through gateway targets, AgentCore Gateway prefixes each tool name with the target name to ensure uniqueness across multiple targets. This is important to understand when building your application logic.

**Naming Pattern:**

**Example:**

If your target is named `my-lambda-target` and provides a tool called `calculate_price`, agents will discover and invoke it as `my-lambda-target__calculate_price`.

**Important Considerations:**

- **For Lambda Targets**: Your Lambda handler must strip the target name prefix before processing the tool request. The full tool name (with prefix) is sent in the event.
- **For MCP Server Targets**: The MCP server receives tool calls with the prefixed name from the gateway.
- **For OpenAPI/Smithy Targets**: The gateway handles the prefix automatically when mapping to API operations based on the `operationId`.

This naming convention ensures that:

- Tools from different targets don't collide even if they have the same name
- Agents can access tools from multiple targets through a single gateway
- Tool names remain unique in the unified tool catalog

For more details, see the [Gateway Tool Naming Documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-tool-naming.html).

### Tools schema For Lambda target

The lambda target need tools schema to understand the fuunction lambda provides. You can upload the tool schema by following 3 ways:

- From a local asset file

```typescript
const toolSchema = agentcore.ToolSchema.fromLocalAsset(
    path.join(__dirname, "schemas", "my-tool-schema.json")
  );
```

- From an existing S3 file:

```typescript

const toolSchema = agentcore.ToolSchema.fromS3File(
    s3.Bucket.fromBucketName(this, "SchemasBucket", "my-schemas-bucket"),
    "tools/complex-tool-schema.json",
    "123456789012"
  );
```

- From Inline:

```typescript
const toolSchema = agentcore.ToolSchema.fromInline([{
      name: "hello_world",
      description: "A simple hello world tool",
      inputSchema: {
        type: agentcore.SchemaDefinitionType.OBJECT,
        properties: {
          name: {
            type: agentcore.SchemaDefinitionType.STRING,
            description: "The name to greet",
          },
        },
        required: ["name"],
      },
    }]);

```

### Api schema For OpenAPI and Smithy target

The OpenAPI and Smithy target need API Schema. The Gateway construct provide three ways to upload API schema for your target:

- From a local asset file (requires binding to scope):

```typescript fixture=default
// When using ApiSchema.fromLocalAsset, you must bind the schema to a scope
const schema = agentcore.ApiSchema.fromLocalAsset(path.join(__dirname, "mySchema.yml"));

schema.bind(this);
```

- From an inline schema:

```typescript fixture=default
const inlineSchema = agentcore.ApiSchema.fromInline(`
openapi: 3.0.3
info:
  title: Library API
  version: 1.0.0
paths:
  /search:
    get:
      summary: Search for books
      operationId: searchBooks
      parameters:
        - name: query
          in: query
          required: true
          schema:
            type: string
`);
```

- From an existing S3 file:

```typescript fixture=default
const bucket = s3.Bucket.fromBucketName(this, "ExistingBucket", "my-schema-bucket");
const s3Schema = agentcore.ApiSchema.fromS3File(bucket, "schemas/action-group.yaml");
```

### Outbound auth

Outbound authorization lets Amazon Bedrock AgentCore gateways securely access gateway targets on behalf of users authenticated
and authorized during Inbound Auth.

AgentCore Gateway supports the following types of outbound authorization:

**IAM-based outbound authorization** – The gateway uses its execution role to authenticate with AWS services. This is the default and most common approach for Lambda targets and AWS service integrations. Use `GatewayCredentialProvider.fromIamRole()`; by default the gateway infers the SigV4 signing service and region from the target endpoint. For **MCP Server** and **OpenAPI** targets, you can override the service, and optionally the region too — useful for cross-region calls or when the service can't be inferred from the URL:

```typescript fixture=default
agentcore.GatewayCredentialProvider.fromIamRole({
  service: 'bedrock-runtime', // SigV4 signing name (typically the endpoint prefix); see the AWS service authorization reference
  region: 'us-east-1',         // defaults to the gateway's region
});
```

The Bedrock AgentCore service only accepts `IamCredentialProvider` with explicit `service` / `region` for MCP Server and OpenAPI targets. Lambda, API Gateway and Smithy targets must use the bare `GatewayCredentialProvider.fromIamRole()` (with no arguments); the CDK enforces this with a synth-time validation.

**2-legged OAuth (OAuth 2LO)** – Use OAuth 2.0 two-legged flow (2LO) for targets that require OAuth authentication.
The gateway authenticates on its own behalf, not on behalf of a user.

**API key** – Use the AgentCore service/AWS console to generate an API key to authenticate access to the gateway target.

**Note > You need to set up the outbound identity before you can create a gateway target.

#### Token Vault credential providers

AgentCore stores outbound **API key** and **OAuth2** client credentials in Token Vault. This module includes L2 constructs that create those resources and connect them to gateway targets.

**Shared OAuth2 fields** — Every `OAuth2CredentialProvider` factory accepts the same **`clientId`** and **`clientSecret`**, plus optional **`oAuth2CredentialProviderName`** and **`tags`**. Extra properties appear only when an IdP needs them (for example **`tenantId`** for Microsoft, or **`issuer`** / endpoint overrides for Okta and other *included* configurations).

**Vendor factories** — Prefer `OAuth2CredentialProvider.usingSlack`, `.usingGithub`, `.usingGoogle`, `.usingMicrosoft`, `.usingOkta`, `.usingAuth0`, `.usingCognito`, and the other `using*` helpers for known providers. Each maps to the matching CloudFormation *included* provider configuration.

**Custom OAuth2 (`usingCustom`)** — Supply **exactly one** of:

- **`discoveryUrl`** — OIDC/OAuth2 discovery document URL (for example `https://idp.example.com/.well-known/openid-configuration`), or
- **`authorizationServerMetadata`** — Manual authorization server metadata (`issuer`, `authorizationEndpoint`, `tokenEndpoint`, and other fields supported by the `AWS::BedrockAgentCore::OAuth2CredentialProvider` resource).

Do not pass both. The construct validates this at synthesis time when values are known; if you use CDK tokens, ensure the resolved template still satisfies the service rules.

**Wiring to gateway targets** — After you create a provider in CDK, pass the construct to **`GatewayCredentialProvider.fromOauthIdentity()`** or **`fromApiKeyIdentity()`** (optional API key header/query settings go in the second argument for API keys). Alternatively, call **`bindForGatewayOAuthTarget`** / **`bindForGatewayApiKeyTarget`** on the provider and pass that object to **`fromOauthIdentityArn`** / **`fromApiKeyIdentityArn`**. You can still pass raw ARNs from the console or API when the provider already exists.

**Example: GitHub OAuth2 and an MCP target**

```typescript fixture=default
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
});

const oauth = agentcore.OAuth2CredentialProvider.usingGithub(this, "GhOAuth", {
  oAuth2CredentialProviderName: "github-oauth",
  clientId: "your-client-id",
  clientSecret: cdk.SecretValue.unsafePlainText("your-client-secret"),
});

gateway.addMcpServerTarget("Mcp", {
  gatewayTargetName: "mcp-server",
  description: "MCP with GitHub OAuth",
  endpoint: "https://my-mcp-server.example.com",
  credentialProviderConfigurations: [
    agentcore.GatewayCredentialProvider.fromOauthIdentity(oauth, {
      scopes: ["read:user"],
    }),
  ],
});
```

**Example: custom IdP with a discovery URL**

```typescript fixture=default
agentcore.OAuth2CredentialProvider.usingCustom(this, "CustomOAuth", {
  oAuth2CredentialProviderName: "custom-idp",
  clientId: "your-client-id",
  clientSecret: cdk.SecretValue.unsafePlainText("your-client-secret"),
  discoveryUrl: "https://idp.example.com/.well-known/openid-configuration",
});
```

**Example: custom IdP with explicit authorization server metadata**

```typescript fixture=default
agentcore.OAuth2CredentialProvider.usingCustom(this, "CustomOAuthMeta", {
  clientId: "your-client-id",
  clientSecret: cdk.SecretValue.unsafePlainText("your-client-secret"),
  authorizationServerMetadata: {
    issuer: "https://idp.example.com",
    authorizationEndpoint: "https://idp.example.com/oauth2/authorize",
    tokenEndpoint: "https://idp.example.com/oauth2/token",
  },
});
```

#### Workload identities

A **workload identity** is the stable identity of an agent in your AWS account within the AgentCore Identity model. It ties together IAM roles, OAuth2 flows, API keys, and workload access tokens so agents can authenticate consistently across environments. For conceptual background, see [Understanding workload identities](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/understanding-agent-identities.html).

**Agent identity directory** — Each account has a single logical directory that holds every workload identity, whether it was created by AgentCore Runtime, AgentCore Gateway, or manually (for example through CloudFormation or the control-plane API). The directory is created automatically when the first workload identity exists. Resource ARNs follow the pattern described in [Understanding the agent identity directory](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agent-identity-directory.html) (`workload-identity-directory/default` and child `workload-identity/<name>` entries).

**When to create one in CDK** — Runtime and Gateway can create workload identities for you during deployment. Use the **`WorkloadIdentity`** L2 when you need a **manually defined** identity (custom name, allowed OAuth2 return URLs, tags) or when integrating workloads that are not driven by those services.

**Construct** — `WorkloadIdentity` maps to **`AWS::BedrockAgentCore::WorkloadIdentity`**. It exposes `workloadIdentityArn`, `workloadIdentityName`, and `workloadIdentityRef` for wiring into IAM or other AgentCore resources. Import an existing identity with **`WorkloadIdentity.fromWorkloadIdentityAttributes`**. Grant helpers (`grantRead`, `grantAdmin`, `grantFullAccess`) align with [directory-level IAM patterns](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agent-identity-directory.html#directory-access-control-and-permissions) such as listing identities on the directory resource and scoping mutations to specific identity ARNs.

**Example**

```typescript fixture=default
new agentcore.WorkloadIdentity(this, "MyWorkloadIdentity", {
  workloadIdentityName: "customer-support-agent-prod",
  allowedResourceOauth2ReturnUrls: ["https://app.example.com/oauth/callback"],
  tags: { team: "agents", env: "prod" },
});
```

### Basic Gateway Target Creation

You can create targets in two ways: using the static factory methods on `GatewayTarget` or using the convenient `addTarget` methods on the gateway instance.

#### Using addTarget methods (Recommended)

This approach is recommended for most use cases, especially when creating targets alongside the gateway. It provides a cleaner, more fluent API by eliminating the need to explicitly pass the gateway reference.

Below are the examples on how you can create Lambda, Smithy, OpenAPI, MCP Server, and API Gateway targets using `addTarget` methods.

```typescript fixture=default
// Create a gateway first
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
});

const lambdaFunction = new lambda.Function(this, "MyFunction", {
  runtime: lambda.Runtime.NODEJS_22_X,
  handler: "index.handler",
  code: lambda.Code.fromInline(`
    exports.handler = async (event) => {
      return {
        statusCode: 200,
        body: JSON.stringify({ message: 'Hello from Lambda!' })
      };
    };
  `),
});

const lambdaTarget = gateway.addLambdaTarget("MyLambdaTarget", {
  gatewayTargetName: "my-lambda-target",
  description: "Lambda function target",
  lambdaFunction: lambdaFunction,
  toolSchema: agentcore.ToolSchema.fromInline([
    {
      name: "hello_world",
      description: "A simple hello world tool",
      inputSchema: {
        type: agentcore.SchemaDefinitionType.OBJECT,
        properties: {
          name: {
            type: agentcore.SchemaDefinitionType.STRING,
            description: "The name to greet",
          },
        },
        required: ["name"],
      },
    },
  ]),
});
```

- OpenAPI Target (using Token Vault L2 construct — preferred)

``` typescript
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
});

// Create an API key credential provider in Token Vault
const apiKeyProvider = new agentcore.ApiKeyCredentialProvider(this, "MyApiKeyProvider", {
  apiKeyCredentialProviderName: "my-apikey",
});

const bucket = s3.Bucket.fromBucketName(this, "ExistingBucket", "my-schema-bucket");
const s3mySchema = agentcore.ApiSchema.fromS3File(bucket, "schemas/myschema.yaml");

// Add an OpenAPI target using the L2 construct directly
const target = gateway.addOpenApiTarget("MyTarget", {
  gatewayTargetName: "my-api-target",
  description: "Target for external API integration",
  apiSchema: s3mySchema,
  credentialProviderConfigurations: [
    agentcore.GatewayCredentialProvider.fromApiKeyIdentity(apiKeyProvider, {
      credentialLocation: agentcore.ApiKeyCredentialLocation.header({
        credentialParameterName: "X-API-Key",
      }),
    }),
  ],
});

// This makes sure your s3 bucket is available before target
target.node.addDependency(bucket);
```

- OpenAPI Target (using ARNs — for providers created outside of CDK)

``` typescript
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
});

// ARNs from the console/API, or from ApiKeyCredentialProvider + bindForGatewayApiKeyTarget
const apiKeyProviderArn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:token-vault/abc123/apikeycredentialprovider/my-apikey"
const apiKeySecretArn = "arn:aws:secretsmanager:us-east-1:123456789012:secret:my-apikey-secret-abc123"

const bucket = s3.Bucket.fromBucketName(this, "ExistingBucket", "my-schema-bucket");
const s3mySchema = agentcore.ApiSchema.fromS3File(bucket, "schemas/myschema.yaml");

// Add an OpenAPI target using ARNs directly
const target = gateway.addOpenApiTarget("MyTarget", {
  gatewayTargetName: "my-api-target",
  description: "Target for external API integration",
  apiSchema: s3mySchema,
  credentialProviderConfigurations: [
    agentcore.GatewayCredentialProvider.fromApiKeyIdentityArn({
      providerArn: apiKeyProviderArn,
      secretArn: apiKeySecretArn, 
      credentialLocation: agentcore.ApiKeyCredentialLocation.header({
        credentialParameterName: "X-API-Key",
      }),
    }),
  ],
});

// This makes sure your s3 bucket is available before target
target.node.addDependency(bucket);
```

- Smithy Target

```typescript fixture=default

const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
});

const smithySchema = agentcore.ApiSchema.fromLocalAsset(
  path.join(__dirname, "models", "smithy-model.json")
);
smithySchema.bind(this);

const smithyTarget = gateway.addSmithyTarget("MySmithyTarget", {
  gatewayTargetName: "my-smithy-target",
  description: "Smithy model target",
  smithyModel: smithySchema,

});
```

- MCP Server Target

```typescript fixture=default
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
});

// OAuth2 (recommended): use OAuth2CredentialProvider + bindForGatewayOAuthTarget, or ARNs from console/API
const oauthProviderArn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:token-vault/abc123/oauth2credentialprovider/my-oauth";
const oauthSecretArn = "arn:aws:secretsmanager:us-east-1:123456789012:secret:my-oauth-secret-abc123";

// Add an MCP server target directly to the gateway
const mcpTarget = gateway.addMcpServerTarget("MyMcpServer", {
  gatewayTargetName: "my-mcp-server",
  description: "External MCP server integration",
  endpoint: "https://my-mcp-server.example.com",
  credentialProviderConfigurations: [
    agentcore.GatewayCredentialProvider.fromOauthIdentityArn({
      providerArn: oauthProviderArn,
      secretArn: oauthSecretArn,
      scopes:['mcp-runtime-server/invoke']
    }),
  ],
});

// Grant sync permission to a Lambda function that will trigger synchronization
const syncFunction = new lambda.Function(this, "SyncFunction", {
  runtime: lambda.Runtime.PYTHON_3_12,
  handler: "index.handler",
  code: lambda.Code.fromInline(`
import boto3

def handler(event, context):
    client = boto3.client('bedrock-agentcore')
    response = client.synchronize_gateway_targets(
        gatewayIdentifier=event['gatewayId'],
        targetIds=[event['targetId']]
    )
    return response
  `),
});

mcpTarget.grantSync(syncFunction);
```

- API Gateway Target

```typescript fixture=default
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
});

const api = new apigateway.RestApi(this, 'MyApi', {
  restApiName: 'my-api',
});

// Uses IAM authorization for outbound auth by default
const apiGatewayTarget = gateway.addApiGatewayTarget("MyApiGatewayTarget", {
  restApi: api,
  apiGatewayToolConfiguration: {
    toolFilters: [
      {
        filterPath: "/pets/*",
        methods: [agentcore.ApiGatewayHttpMethod.GET],
      },
    ],
  },
});
```

#### Using static factory methods

Use static factory methods when working with imported gateways, creating targets in different constructs/stacks, or when you need more explicit control over the construct tree hierarchy.

Create Gateway target using static convenience methods.

- Lambda Target

```typescript fixture=default
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
});

const lambdaFunction = new lambda.Function(this, "MyFunction", {
  runtime: lambda.Runtime.NODEJS_22_X,
  handler: "index.handler",
  code: lambda.Code.fromInline(`
        exports.handler = async (event) => {
            return {
                statusCode: 200,
                body: JSON.stringify({ message: 'Hello from Lambda!' })
            };
        };
    `),
});

// Create a gateway target with Lambda and tool schema 
const target = agentcore.GatewayTarget.forLambda(this, "MyLambdaTarget", {
  gatewayTargetName: "my-lambda-target",
  description: "Target for Lambda function integration",
  gateway: gateway,
  lambdaFunction: lambdaFunction,
  toolSchema: agentcore.ToolSchema.fromLocalAsset(
    path.join(__dirname, "schemas", "my-tool-schema.json")
  ),
});
```

- OpenAPI Target

```typescript fixture=default

const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
});

// Outbound auth: ApiKeyCredentialProvider + bindForGatewayApiKeyTarget, or ARNs from console/API
const apiKeyIdentityArn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:token-vault/abc123/apikeycredentialprovider/my-apikey"
const apiKeySecretArn = "arn:aws:secretsmanager:us-east-1:123456789012:secret:my-apikey-secret-abc123"

const opneapiSchema = agentcore.ApiSchema.fromLocalAsset(path.join(__dirname, "mySchema.yml"));
opneapiSchema.bind(this);

// Create a gateway target with OpenAPI Schema 
const target = agentcore.GatewayTarget.forOpenApi(this, "MyTarget", {
  gatewayTargetName: "my-api-target",
  description: "Target for external API integration",
  gateway: gateway,  // Note: you need to pass the gateway reference
  apiSchema: opneapiSchema,
  credentialProviderConfigurations: [
    agentcore.GatewayCredentialProvider.fromApiKeyIdentityArn({
     providerArn: apiKeyIdentityArn,
     secretArn: apiKeySecretArn
    }),
  ],
});

```

- Smithy Target

```typescript fixture=default

const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
});

const smithySchema = agentcore.ApiSchema.fromLocalAsset(
  path.join(__dirname, "models", "smithy-model.json")
);
smithySchema.bind(this);

// Create a gateway target with Smithy Model and OAuth 
const target = agentcore.GatewayTarget.forSmithy(this, "MySmithyTarget", {
  gatewayTargetName: "my-smithy-target",
  description: "Target for Smithy model integration",
  gateway: gateway,
  smithyModel: smithySchema,
});

```

- MCP Server Target

```typescript fixture=default
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
});

// OAuth2 (recommended): use OAuth2CredentialProvider + bindForGatewayOAuthTarget, or ARNs from console/API
const oauthProviderArn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:token-vault/abc123/oauth2credentialprovider/my-oauth";
const oauthSecretArn = "arn:aws:secretsmanager:us-east-1:123456789012:secret:my-oauth-secret-abc123";

// Create a gateway target with MCP Server
const mcpTarget = agentcore.GatewayTarget.forMcpServer(this, "MyMcpServer", {
  gatewayTargetName: "my-mcp-server",
  description: "External MCP server integration",
  gateway: gateway,
  endpoint: "https://my-mcp-server.example.com",
  credentialProviderConfigurations: [
    agentcore.GatewayCredentialProvider.fromOauthIdentityArn({
      providerArn: oauthProviderArn,
      secretArn: oauthSecretArn,
      scopes:['mcp-runtime-server/invoke']
    }),
  ],
});
```

- API Gateway Target

```typescript fixture=default
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
});

const api = new apigateway.RestApi(this, 'MyApi', {
  restApiName: 'my-api',
});

// Create a gateway target using the static factory method
const apiGatewayTarget = agentcore.GatewayTarget.forApiGateway(this, "MyApiGatewayTarget", {
  gatewayTargetName: "my-api-gateway-target",
  description: "Target for API Gateway REST API integration",
  gateway: gateway,
  restApi: api,
  apiGatewayToolConfiguration: {
    toolFilters: [
      {
        filterPath: "/pets/*",
        methods: [agentcore.ApiGatewayHttpMethod.GET, agentcore.ApiGatewayHttpMethod.POST],
      },
    ],
  },
  metadataConfiguration: {
    allowedRequestHeaders: ["X-User-Id"],
    allowedQueryParameters: ["limit"],
  },
});
```

### Advanced Usage: Direct Configuration for gateway target

For advanced use cases where you need full control over the target configuration, you can create configurations manually using the static factory methods and use the GatewayTarget constructor directly.

#### Configuration Factory Methods

Each target type has a corresponding configuration class with a static `create()` method:

- **Lambda**: `LambdaTargetConfiguration.create(lambdaFunction, toolSchema)`
- **OpenAPI**: `OpenApiTargetConfiguration.create(apiSchema, validateSchema?)`
- **Smithy**: `SmithyTargetConfiguration.create(smithyModel)`
- **API Gateway**: `ApiGatewayTargetConfiguration.create(props)`

#### Example: Lambda Target with Custom Configuration

```typescript
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
});

const myLambdaFunction = new lambda.Function(this, "MyFunction", {
  runtime: lambda.Runtime.NODEJS_22_X,
  handler: "index.handler",
  code: lambda.Code.fromInline(`
    exports.handler = async (event) => ({ statusCode: 200 });
  `),
});

const myToolSchema = agentcore.ToolSchema.fromInline([{
  name: "my_tool",
  description: "My custom tool",
  inputSchema: {
    type: agentcore.SchemaDefinitionType.OBJECT,
    properties: {},
  },
}]);

// Create a custom Lambda configuration
const customConfig = agentcore.LambdaTargetConfiguration.create(
  myLambdaFunction,
  myToolSchema
);

// Use the GatewayTarget constructor directly
const target = new agentcore.GatewayTarget(this, "AdvancedTarget", {
  gateway: gateway,
  gatewayTargetName: "advanced-target",
  targetConfiguration: customConfig,  // Manually created configuration
  credentialProviderConfigurations: [
    agentcore.GatewayCredentialProvider.fromIamRole()
  ]
});
```

This approach gives you full control over the configuration but is typically not necessary for most use cases. The convenience methods (`GatewayTarget.forLambda()`, `GatewayTarget.forOpenApi()`, `GatewayTarget.forSmithy()`, `GatewayTarget.forApiGateway()`) handle all of this internally.

### Gateway Interceptors

Gateway interceptors allow you to run custom code during each gateway invocation to implement fine-grained access control, transform requests and responses, or implement custom authorization logic. A gateway can have at most one REQUEST interceptor and one RESPONSE interceptor.

**Interceptor Types:**

- **REQUEST interceptors**: Execute before the gateway calls the target. Useful for request validation, transformation, or custom authorization
- **RESPONSE interceptors**: Execute after the target responds but before the gateway sends the response back. Useful for response transformation, filtering, or adding custom headers

**Security Best Practices:**

1. Keep `passRequestHeaders` disabled unless absolutely necessary (default: false)
2. Implement idempotent Lambda functions (gateway may retry on failures)
3. Restrict gateway execution role to specific Lambda functions
4. Avoid logging sensitive information in your interceptor

For more information, see the [Gateway Interceptors documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-interceptors.html).

#### Adding Interceptors via Constructor

```typescript fixture=default
// Create Lambda functions for interceptors
const requestInterceptorFn = new lambda.Function(this, "RequestInterceptor", {
  runtime: lambda.Runtime.PYTHON_3_12,
  handler: "index.handler",
  code: lambda.Code.fromInline(`
def handler(event, context):
    # Validate and transform request
    return {
        "interceptorOutputVersion": "1.0",
        "mcp": {
            "transformedGatewayRequest": event["mcp"]["gatewayRequest"]
        }
    }
  `),
});

const responseInterceptorFn = new lambda.Function(this, "ResponseInterceptor", {
  runtime: lambda.Runtime.PYTHON_3_12,
  handler: "index.handler",
  code: lambda.Code.fromInline(`
def handler(event, context):
    # Filter or transform response
    return {
        "interceptorOutputVersion": "1.0",
        "mcp": {
            "transformedGatewayResponse": event["mcp"]["gatewayResponse"]
        }
    }
  `),
});

// Create gateway with interceptors
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
  interceptorConfigurations: [
    agentcore.LambdaInterceptor.forRequest(requestInterceptorFn, {
      passRequestHeaders: true  // Only if you need to inspect headers
    }),
    agentcore.LambdaInterceptor.forResponse(responseInterceptorFn)
  ]
});
```

**Automatic Permission Granting:**

When you add a Lambda interceptor to a gateway (either via constructor or `addInterceptor()`), the gateway's IAM role automatically receives `lambda:InvokeFunction` permission on the Lambda function. This permission grant happens internally during the bind process - you do not need to manually configure these IAM permissions.

#### Adding Interceptors Dynamically

```typescript fixture=default
// Create a gateway first
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
});

// Create Lambda functions for interceptors
const requestInterceptorFn = new lambda.Function(this, "RequestInterceptor", {
  runtime: lambda.Runtime.PYTHON_3_12,
  handler: "index.handler",
  code: lambda.Code.fromInline(`
def handler(event, context):
    # Custom request validation logic
    return {
        "interceptorOutputVersion": "1.0",
        "mcp": {
            "transformedGatewayRequest": event["mcp"]["gatewayRequest"]
        }
    }
  `),
});

const responseInterceptorFn = new lambda.Function(this, "ResponseInterceptor", {
  runtime: lambda.Runtime.PYTHON_3_12,
  handler: "index.handler",
  code: lambda.Code.fromInline(`
def handler(event, context):
    # Filter sensitive data from response
    return {
        "interceptorOutputVersion": "1.0",
        "mcp": {
            "transformedGatewayResponse": event["mcp"]["gatewayResponse"]
        }
    }
  `),
});

gateway.addInterceptor(
  agentcore.LambdaInterceptor.forRequest(requestInterceptorFn, {
    passRequestHeaders: false  // Default, headers not passed for security
  })
);

gateway.addInterceptor(
  agentcore.LambdaInterceptor.forResponse(responseInterceptorFn)
);
```

### Gateway Target IAM Permissions

The Gateway Target construct provides convenient methods for granting IAM permissions:

```typescript fixture=default
// Create a gateway and target
const gateway = new agentcore.Gateway(this, "MyGateway", {
  gatewayName: "my-gateway",
});

const smithySchema = agentcore.ApiSchema.fromLocalAsset(
  path.join(__dirname, "models", "smithy-model.json")
);
smithySchema.bind(this);

// Create a gateway target with Smithy Model and OAuth 
const target = agentcore.GatewayTarget.forSmithy(this, "MySmithyTarget", {
  gatewayTargetName: "my-smithy-target",
  description: "Target for Smithy model integration",
  gateway: gateway,
  smithyModel: smithySchema,
});

// Create a role that needs access to the gateway target
const userRole = new iam.Role(this, "UserRole", {
  assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
});

// Grant read permissions (Get and List actions)
target.grantRead(userRole);

// Grant manage permissions (Create, Update, Delete actions)
target.grantManage(userRole);

// Grant specific custom permissions
target.grant(userRole, "bedrock-agentcore:GetGatewayTarget");


// Grants permission to invoke this Gateway
gateway.grantInvoke(userRole);
```

## Memory

Memory is a critical component of intelligence. While Large Language Models (LLMs) have impressive capabilities, they lack persistent memory across conversations. Amazon Bedrock AgentCore Memory addresses this limitation by providing a managed service that enables AI agents to maintain context over time, remember important facts, and deliver consistent, personalized experiences.

AgentCore Memory operates on two levels:

- **Short-Term Memory**: Immediate conversation context and session-based information that provides continuity within a single interaction or closely related sessions.
- **Long-Term Memory**: Persistent information extracted and stored across multiple conversations, including facts, preferences, and summaries that enable personalized experiences over time.

When you interact with the memory via the `CreateEvent` API, you store interactions in Short-Term Memory (STM) instantly. These interactions can include everything from user messages, assistant responses, to tool actions.

To write to long-term memory, you need to configure extraction strategies which define how and where to store information from conversations for future use. These strategies are asynchronously processed from raw events after every few turns based on the strategy that was selected. You can't create long term memory records directly, as they are extracted asynchronously by AgentCore Memory.

### Memory Properties

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `memoryName` | `string` | No | The name of the memory. If not provided, a unique name will be auto-generated |
| `expirationDuration` | `Duration` | No | Short-term memory expiration in days (between 7 and 365). Default: 90 days |
| `description` | `string` | No | Optional description for the memory. Default: no description. |
| `kmsKey` | `IKey` | No | Custom KMS key to use for encryption. Default: Your data is encrypted with a key that AWS owns and manages for you |
| `memoryStrategies` | `MemoryStrategyBase[]` | No | Built-in extraction strategies to use for this memory. Default: No extraction strategies (short term memory only) |
| `executionRole` | `iam.IRole` | No | The IAM role that provides permissions for the memory to access AWS services. Default: A new role will be created. |
| `tags` | `{ [key: string]: string }` | No | Tags for memory. Default: no tags. |

### Basic Memory Creation

Below you can find how to configure a simple short-term memory (STM) with no long-term memory extraction strategies.
Note how you set `expirationDuration`, which defines the time the events will be stored in the short-term memory before they expire.

```typescript fixture=default

// Create a basic memory with default settings, no LTM strategies
const memory = new agentcore.Memory(this, "MyMemory", {
  memoryName: "my_memory",
  description: "A memory for storing user interactions for a period of 90 days",
  expirationDuration: cdk.Duration.days(90),
});
```

Basic Memory with Custom KMS Encryption

```typescript fixture=default
// Create a custom KMS key for encryption
const encryptionKey = new kms.Key(this, "MemoryEncryptionKey", {
  enableKeyRotation: true,
  description: "KMS key for memory encryption",
});

// Create memory with custom encryption
const memory = new agentcore.Memory(this, "MyMemory", {
  memoryName: "my_encrypted_memory",
  description: "Memory with custom KMS encryption",
  expirationDuration: cdk.Duration.days(90),
  kmsKey: encryptionKey,
});
```

### LTM Memory Extraction Stategies

If you need long-term memory for context recall across sessions, you can setup memory extraction strategies
to extract the relevant memory from the raw events.

Amazon Bedrock AgentCore Memory has different memory strategies for extracting and organizing information:

- **Summarization**: to summarize interactions to preserve critical context and key insights.
- **Semantic Memory**: to extract general factual knowledge, concepts and meanings from raw conversations using vector embeddings.
This enables similarity-based retrieval of relevant facts and context.
- **User Preferences**: to extract user behavior patterns from raw conversations.

You can use built-in extraction strategies for quick setup, or create custom extraction strategies with specific models and prompt templates.

### Memory with Built-in Strategies

The library provides four built-in LTM strategies. These are default strategies for organizing and extracting memory data,
each optimized for specific use cases.

For example: An agent helps multiple users with cloud storage setup. From these conversations,
see how each strategy processes users expressing confusion about account connection:

1. **Summarization Strategy** (`MemoryStrategy.usingBuiltInSummarization()`)
This strategy compresses conversations into concise overviews, preserving essential context and key insights for quick recall.
Extracted memory example: Users confused by cloud setup during onboarding.

   - Extracts concise summaries to preserve critical context and key insights
   - Namespace: `/strategies/{memoryStrategyId}/actors/{actorId}/sessions/{sessionId}`

2. **Semantic Memory Strategy** (`MemoryStrategy.usingBuiltInSemantic()`)
Distills general facts, concepts, and underlying meanings from raw conversational data, presenting the information in a context-independent format.
Extracted memory example: In-context learning = task-solving via examples, no training needed.

   - Extracts general factual knowledge, concepts and meanings from raw conversations
   - Namespace: `/strategies/{memoryStrategyId}/actors/{actorId}`

3. **User Preference Strategy** (`MemoryStrategy.usingBuiltInUserPreference()`)
Captures individual preferences, interaction patterns, and personalized settings to enhance future experiences.
Extracted memory example: User needs clear guidance on cloud storage account connection during onboarding.

   - Extracts user behavior patterns from raw conversations
   - Namespace: `/strategies/{memoryStrategyId}/actors/{actorId}`

4. **Episodic Memory Strategy** (`MemoryStrategy.usingBuiltInEpisodic()`)
Captures meaningful slices of user and system interactions, preserve them into compact records after summarizing.
Extracted memory example: User first asked about pricing on Monday, then requested feature comparison on Tuesday, finally made purchase decision on Wednesday.

   - Captures event sequences and temporal relationships
   - Namespace: `/strategy/{memoryStrategyId}/actor/{actorId}/session/{sessionId}`
   - Reflections: `/strategy/{memoryStrategyId}/actor/{actorId}` 

```typescript fixture=default
// Create memory with built-in strategies
const memory = new agentcore.Memory(this, "MyMemory", {
  memoryName: "my_memory",
  description: "Memory with built-in strategies",
  expirationDuration: cdk.Duration.days(90),
  memoryStrategies: [
    agentcore.MemoryStrategy.usingBuiltInSummarization(),
    agentcore.MemoryStrategy.usingBuiltInSemantic(),
    agentcore.MemoryStrategy.usingBuiltInUserPreference(),
    agentcore.MemoryStrategy.usingBuiltInEpisodic(),
  ],
});
```

The name generated for each built in memory strategy is as follows:

- For Summarization: `summary_builtin_cdk001`
- For Semantic:`semantic_builtin_cdk001>`
- For User Preferences: `preference_builtin_cdk001`
- For Episodic : `episodic_builtin_cdkGen0001`

### Memory with custom Strategies

With Long-Term Memory, organization is managed through Namespaces.

An `actor` refers to entity such as end users or agent/user combinations. For example, in a coding support chatbot,
the actor is usually the developer asking questions. Using the actor ID helps the system know which user the memory belongs to,
keeping each user's data separate and organized.

A `session` is usually a single conversation or interaction period between the user and the AI agent.
It groups all related messages and events that happen during that conversation.

A `namespace` is used to logically group and organize long-term memories. It ensures data stays neat, separate, and secure.

With AgentCore Memory, you need to add a namespace when you define a memory strategy. This namespace helps define where the long-term memory
will be logically grouped. Every time a new long-term memory is extracted using this memory strategy, it is saved under the namespace you set.
This means that all long-term memories are scoped to their specific namespace, keeping them organized and preventing any mix-ups with other
users or sessions. You should use a hierarchical format separated by forward slashes /. This helps keep memories organized clearly. As needed,
you can choose to use the below pre-defined variables within braces in the namespace based on your applications' organization needs:

- `actorId` – Identifies who the long-term memory belongs to, such as a user
- `memoryStrategyId` – Shows which memory strategy is being used. This strategy identifier is auto-generated when you create a memory using CreateMemory operation.
- `sessionId` – Identifies which session or conversation the memory is from.

For example, if you define the following namespace as the input to your strategy in CreateMemory operation:

```shell
/strategy/{memoryStrategyId}/actor/{actorId}/session/{sessionId}
```

After memory creation, this namespace might look like:

```shell
/strategy/summarization-93483043//actor/actor-9830m2w3/session/session-9330sds8
```

You can customise the namespace, i.e. where the memories are stored by using the following methods:

1. **Summarization Strategy** (`MemoryStrategy.usingSummarization(props)`)
1. **Semantic Memory Strategy** (`MemoryStrategy.usingSemantic(props)`)
1. **User Preference Strategy** (`MemoryStrategy.usingUserPreference(props)`)
1. **Episodic Memory Strategy** (`MemoryStrategy.usingEpisodic(props)`)

```typescript fixture=default
// Create memory with custom strategies
const memory = new agentcore.Memory(this, "MyMemory", {
  memoryName: "my_memory",
  description: "Memory with custom strategies",
  expirationDuration: cdk.Duration.days(90),
  memoryStrategies: [
    agentcore.MemoryStrategy.usingUserPreference({
        strategyName: "CustomerPreferences",
        namespaces: ["support/customer/{actorId}/preferences"]
    }),
    agentcore.MemoryStrategy.usingSemantic({
        strategyName: "CustomerSupportSemantic",
        namespaces: ["support/customer/{actorId}/semantic"]
    }),
     agentcore.MemoryStrategy.usingEpisodic({
        strategyName: "customerJourneyEpisodic",
        namespaces: ["/journey/customer/{actorId}/episodes"],
        reflectionConfiguration: {
            namespaces: ["/journey/customer/{actorId}/reflections"]
        }
    }),
  ],
});
```

Custom memory strategies let you tailor memory extraction and consolidation to your specific domain or use case.
You can override the prompts for extracting and consolidating semantic, summary, or user preferences.
You can also choose the model that you want to use for extraction and consolidation.

The custom prompts you create are appended to a non-editable system prompt.

Since a custom strategy requires you to invoke certain FMs, you need a role with appropriate permissions. For that, you can:

- Let the L2 construct create a minimum permission role for you when use L2 Bedrock Foundation Models.
- Use a custom role with the overly permissive `AmazonBedrockAgentCoreMemoryBedrockModelInferenceExecutionRolePolicy` managed policy.
- Use a custom role with your own custom policies.

#### Memory with Custom Execution Role

Keep in mind that memories that **do not** use custom strategies do not require a service role.
So even if you provide it, it will be ignored as it will never be used.

```typescript fixture=default
// Create a custom execution role
const executionRole = new iam.Role(this, "MemoryExecutionRole", {
  assumedBy: new iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
  managedPolicies: [
    iam.ManagedPolicy.fromAwsManagedPolicyName(
      "AmazonBedrockAgentCoreMemoryBedrockModelInferenceExecutionRolePolicy"
    ),
  ],
});

// Create memory with custom execution role
const memory = new agentcore.Memory(this, "MyMemory", {
  memoryName: "my_memory",
  description: "Memory with custom execution role",
  expirationDuration: cdk.Duration.days(90),
  executionRole: executionRole,
});
```

In customConsolidation and customExtraction, the model property uses `IModel` from `aws-cdk-lib/aws-bedrock`.

```typescript fixture=default
// Create a custom semantic memory strategy
const customSemanticStrategy = agentcore.MemoryStrategy.usingSemantic({
  strategyName: "customSemanticStrategy",
  description: "Custom semantic memory strategy",
  namespaces: ["/custom/strategies/{memoryStrategyId}/actors/{actorId}"],
  customConsolidation: {
    model: bedrock.FoundationModel.fromFoundationModelId(this, 'ConsolidationModel',
      bedrock.FoundationModelIdentifier.ANTHROPIC_CLAUDE_3_5_SONNET_20241022_V2_0,
    ),
    appendToPrompt: "Custom consolidation prompt for semantic memory",
  },
  customExtraction: {
    model: bedrock.FoundationModel.fromFoundationModelId(this, 'ExtractionModel',
      bedrock.FoundationModelIdentifier.ANTHROPIC_CLAUDE_3_5_SONNET_20241022_V2_0,
    ),
    appendToPrompt: "Custom extraction prompt for semantic memory",
  },
});

// Create memory with custom strategy
const memory = new agentcore.Memory(this, "MyMemory", {
  memoryName: "my-custom-memory",
  description: "Memory with custom strategy",
  expirationDuration: cdk.Duration.days(90),
  memoryStrategies: [customSemanticStrategy],
});
```

### Memory with self-managed Strategies

A self-managed strategy in Amazon Bedrock AgentCore Memory gives you complete control over your memory extraction and consolidation pipelines.
With a self-managed strategy, you can build custom memory processing workflows while leveraging Amazon Bedrock AgentCore for storage and retrieval.

For additional information, you can refer to the [developer guide for self managed strategies](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-self-managed-strategies.html).

Create the required AWS resources including:

- an S3 bucket in your account where Amazon Bedrock AgentCore will deliver batched event payloads.
- an SNS topic for job notifications. Use FIFO topics if processing order within sessions is important for your use case.

The construct will apply the correct permissions to the memory execution role to access these resources.

```typescript fixture=default

const bucket = new s3.Bucket(this, 'memoryBucket', {
  bucketName: 'test-memory',
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});

const topic = new sns.Topic(this, 'topic');

// Create a custom semantic memory strategy
const selfManagedStrategy = agentcore.MemoryStrategy.usingSelfManaged({
  strategyName: "selfManagedStrategy",
  description: "self managed memory strategy",
  historicalContextWindowSize: 5,
  invocationConfiguration: {
    topic: topic,
    s3Location: {
      bucketName: bucket.bucketName,
      objectKey: 'memory/',
    }
  },
  triggerConditions: {
    messageBasedTrigger: 1,
    timeBasedTrigger: cdk.Duration.seconds(10),
    tokenBasedTrigger: 100
  }
});

// Create memory with custom strategy
const memory = new agentcore.Memory(this, "MyMemory", {
  memoryName: "my-custom-memory",
  description: "Memory with custom strategy",
  expirationDuration: cdk.Duration.days(90),
  memoryStrategies: [selfManagedStrategy],
});
```

### Memory Strategy Methods

You can add new memory strategies to the memory construct using the `addMemoryStrategy()` method, for instance:

```typescript fixture=default
// Create memory without initial strategies
const memory = new agentcore.Memory(this, "test-memory", {
  memoryName: "test_memory_add_strategy",
  description: "A test memory for testing addMemoryStrategy method",
  expirationDuration: cdk.Duration.days(90),
});

// Add strategies after instantiation
memory.addMemoryStrategy(agentcore.MemoryStrategy.usingBuiltInSummarization());
memory.addMemoryStrategy(agentcore.MemoryStrategy.usingBuiltInSemantic());
```

### Memory with Stream Delivery

You can configure stream delivery resources to enable real-time push-based streaming of memory record lifecycle events (created, updated, deleted) to Amazon Kinesis Data Streams. This allows you to react to memory changes in real-time, build event-driven architectures, or feed memory events into downstream analytics pipelines.

Delivery targets are created with the static factory methods on `StreamDeliveryResource`, one per target type. Kinesis Data Streams is currently the only supported target:

```typescript fixture=default
// Create a Kinesis Data Stream
const stream = new kinesis.Stream(this, 'MemoryEventStream', {
  streamName: 'memory-events',
});

const memory = new agentcore.Memory(this, 'MemoryWithStreamDelivery', {
  memoryName: 'memory_with_stream',
  description: 'Memory with Kinesis stream delivery',
  expirationDuration: cdk.Duration.days(90),
  streamDeliveryResources: [
    agentcore.StreamDeliveryResource.kinesis(stream, {
      contentConfigurations: [
        {
          type: agentcore.StreamDeliveryContentType.MEMORY_RECORDS,
          level: agentcore.StreamDeliveryContentLevel.METADATA_ONLY,
        },
      ],
    }),
  ],
});
```

There is no default content level — you must choose one explicitly. `METADATA_ONLY` delivers only the record ID, timestamps, and event type. `FULL_CONTENT` delivers the complete memory record body, which can contain personally identifiable information and other sensitive conversation content, so make sure the destination stream and its consumers are an appropriate place for that data:

```typescript fixture=default
const stream = new kinesis.Stream(this, 'MemoryEventStream');

const memory = new agentcore.Memory(this, 'MemoryWithStreamDelivery', {
  memoryName: 'memory_with_stream',
  streamDeliveryResources: [
    agentcore.StreamDeliveryResource.kinesis(stream, {
      contentConfigurations: [
        {
          type: agentcore.StreamDeliveryContentType.MEMORY_RECORDS,
          // Streams complete memory record bodies, which may include sensitive data
          level: agentcore.StreamDeliveryContentLevel.FULL_CONTENT,
        },
      ],
    }),
  ],
});
```

You can also add stream delivery resources after instantiation using the `addStreamDeliveryResource()` method:

```typescript fixture=default
const memory = new agentcore.Memory(this, 'MyMemory', {
  memoryName: 'my_memory',
});

const stream = new kinesis.Stream(this, 'EventStream');

memory.addStreamDeliveryResource(agentcore.StreamDeliveryResource.kinesis(stream, {
  contentConfigurations: [
    {
      type: agentcore.StreamDeliveryContentType.MEMORY_RECORDS,
      level: agentcore.StreamDeliveryContentLevel.METADATA_ONLY,
    },
  ],
}));
```

Only one stream delivery resource is currently supported (a CloudFormation maximum); providing more than one fails at synth with `TooManyStreamDeliveryResources`.

The memory execution role is automatically granted write permissions (`kinesis:PutRecord`, `kinesis:PutRecords`, `kinesis:ListShards`, `kinesis:DescribeStream`) to each configured Kinesis stream. If the stream uses a customer-managed KMS key, encryption permissions are also granted automatically.

Encryption permissions can only be granted when the stream's key is known to CDK — that is, for streams you create and for streams imported with `Stream.fromStreamAttributes({ encryptionKey })`. A stream imported with `Stream.fromStreamArn()` carries no key reference, so grant the key permissions yourself in that case.

## Online Evaluation

The Online Evaluation construct enables continuous monitoring and assessment of your agent's performance using live traffic. It automatically samples agent traces from CloudWatch Logs or Agent Endpoints and applies built-in evaluators to assess quality metrics like helpfulness, correctness, and safety.

### Prerequisites

Before creating an `OnlineEvaluationConfig`, ensure the following are configured in your account:

- **CloudWatch Transaction Search** enabled — this creates the `aws/spans` log group required by the evaluation service. See [Enable Transaction Search](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-configure.html).
- **AWS Distro for OpenTelemetry (ADOT) SDK** instrumenting your agent to emit traces.

For full details, see [AgentCore Evaluations Prerequisites](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/evaluations-prerequisites.html).

### Online Evaluation Properties

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `onlineEvaluationConfigName` | `string` | Yes | The name of the online evaluation configuration. Must start with a letter and can contain a-z, A-Z, 0-9, _ (underscore). Maximum 48 characters |
| `evaluators` | `EvaluatorSelector[]` | Yes | The list of built-in evaluators to apply during evaluation. Minimum 1, maximum 10 |
| `dataSource` | `DataSourceConfig` | Yes | The data source configuration specifying where to read agent traces from |
| `executionRole` | `iam.IRole` | No | The IAM role for evaluation. If not provided, a role will be created automatically |
| `description` | `string` | No | Description of the evaluation configuration. Maximum 200 characters |
| `samplingPercentage` | `number` | No | Percentage of traces to sample (0.01-100). Default: 10 |
| `filters` | `FilterConfig[]` | No | Filters to determine which traces to evaluate. Use `FilterValue.string()`, `FilterValue.number()`, or `FilterValue.boolean()` for typed filter values. Maximum 5 |
| `sessionTimeout` | `Duration` | No | Duration of inactivity before a session is considered complete (1-1440 minutes). Default: `Duration.minutes(15)` |
| `tags` | `{ [key: string]: string }` | No | Tags for the evaluation configuration |

### Basic Online Evaluation Creation

Create an online evaluation configuration with built-in evaluators:

```typescript fixture=default
const evaluation = new agentcore.OnlineEvaluationConfig(this, 'MyEvaluation', {
  onlineEvaluationConfigName: 'my_evaluation',
  evaluators: [
    agentcore.EvaluatorSelector.builtin(agentcore.BuiltinEvaluator.HELPFULNESS),
    agentcore.EvaluatorSelector.builtin(agentcore.BuiltinEvaluator.CORRECTNESS),
  ],
  dataSource: agentcore.DataSourceConfig.fromCloudWatchLogs({
    logGroupNames: ['/aws/bedrock-agentcore/my-agent'],
    serviceNames: ['my-agent.default'],
  }),
});
```

### Built-in Evaluators

Amazon Bedrock AgentCore provides 13 built-in evaluators that assess different aspects of agent performance:

**Session-Level Evaluators:**

- `GOAL_SUCCESS_RATE` - Evaluates whether the conversation successfully meets the user's goals

**Trace-Level Evaluators:**

- `HELPFULNESS` - How useful and valuable the agent's response is
- `CORRECTNESS` - Whether the information is factually accurate
- `FAITHFULNESS` - Whether the response is faithful to the provided context
- `HARMFULNESS` - Whether the response contains harmful content
- `STEREOTYPING` - Detects content that makes generalizations about individuals or groups
- `REFUSAL` - Whether the agent appropriately refuses harmful requests
- `COHERENCE` - Whether the response is logically coherent
- `RESPONSE_RELEVANCE` - Whether the response appropriately addresses the user's query
- `CONCISENESS` - Whether the response is appropriately concise
- `INSTRUCTION_FOLLOWING` - How well the agent follows system instructions

**Tool Call-Level Evaluators:**

- `TOOL_SELECTION_ACCURACY` - Whether the agent selected the appropriate tool
- `TOOL_PARAMETER_ACCURACY` - How accurately the agent extracts parameters from user queries

```typescript fixture=default
const evaluation = new agentcore.OnlineEvaluationConfig(this, 'ComprehensiveEval', {
  onlineEvaluationConfigName: 'comprehensive_evaluation',
  evaluators: [
    // Session level
    agentcore.EvaluatorSelector.builtin(agentcore.BuiltinEvaluator.GOAL_SUCCESS_RATE),
    // Trace level - quality
    agentcore.EvaluatorSelector.builtin(agentcore.BuiltinEvaluator.HELPFULNESS),
    agentcore.EvaluatorSelector.builtin(agentcore.BuiltinEvaluator.CORRECTNESS),
    agentcore.EvaluatorSelector.builtin(agentcore.BuiltinEvaluator.COHERENCE),
    // Trace level - safety
    agentcore.EvaluatorSelector.builtin(agentcore.BuiltinEvaluator.HARMFULNESS),
    agentcore.EvaluatorSelector.builtin(agentcore.BuiltinEvaluator.STEREOTYPING),
    // Tool call level
    agentcore.EvaluatorSelector.builtin(agentcore.BuiltinEvaluator.TOOL_SELECTION_ACCURACY),
  ],
  dataSource: agentcore.DataSourceConfig.fromCloudWatchLogs({
    logGroupNames: ['/aws/bedrock-agentcore/my-agent'],
    serviceNames: ['my-agent.default'],
  }),
});
```

### Custom Evaluators

Custom evaluators let you define evaluation logic tailored to your specific use cases. You can create custom evaluators using two strategies:

- **LLM-as-a-Judge**: Uses a foundation model with custom instructions and a rating scale to assess agent performance.
- **Code-based**: Uses a Lambda function for custom evaluation logic.

| Property | Type | Required | Description |
|---|---|---|---|
| `evaluatorName` | `string` | Yes | Name of the evaluator. Must start with a letter, a-z, A-Z, 0-9, _ only. Maximum 48 characters |
| `evaluatorConfig` | `EvaluatorConfig` | Yes | Configuration defining how the evaluator assesses performance |
| `level` | `EvaluationLevel` | Yes | The level at which the evaluator operates: `TOOL_CALL`, `TRACE`, or `SESSION` |
| `description` | `string` | No | Description of the evaluator. Maximum 200 characters |
| `tags` | `{ [key: string]: string }` | No | Tags for the evaluator. A list of key:value pairs to apply to this Evaluator resource |

#### LLM-as-a-Judge Evaluator

Create a custom evaluator that uses a foundation model to assess agent performance:

```typescript fixture=default
// LLM-as-a-Judge with categorical rating scale
const categoricalEvaluator = new agentcore.Evaluator(this, 'CategoricalEvaluator', {
  evaluatorName: 'domain_accuracy_evaluator',
  level: agentcore.EvaluationLevel.SESSION,
  description: 'Evaluates domain-specific accuracy of agent responses',
  evaluatorConfig: agentcore.EvaluatorConfig.llmAsAJudge({
    instructions: 'Evaluate whether the agent response is accurate within the healthcare domain.',
    modelId: 'us.anthropic.claude-sonnet-4-6',
    ratingScale: agentcore.EvaluatorRatingScale.categorical([
      { label: 'Accurate', definition: 'The response contains factually correct healthcare information.' },
      { label: 'Inaccurate', definition: 'The response contains incorrect or misleading healthcare information.' },
    ]),
  }),
});

// LLM-as-a-Judge with numerical rating scale and inference config
const numericalEvaluator = new agentcore.Evaluator(this, 'NumericalEvaluator', {
  evaluatorName: 'response_quality_evaluator',
  level: agentcore.EvaluationLevel.TRACE,
  evaluatorConfig: agentcore.EvaluatorConfig.llmAsAJudge({
    instructions: 'Rate the overall quality of the agent response on a scale of 1 to 5.',
    modelId: 'us.anthropic.claude-sonnet-4-6',
    ratingScale: agentcore.EvaluatorRatingScale.numerical([
      { label: 'Poor', definition: 'Inadequate response.', value: 1 },
      { label: 'Below Average', definition: 'Partially addresses the query.', value: 2 },
      { label: 'Average', definition: 'Adequately addresses the query.', value: 3 },
      { label: 'Good', definition: 'Well-structured and accurate response.', value: 4 },
      { label: 'Excellent', definition: 'Outstanding response exceeding expectations.', value: 5 },
    ]),
    inferenceConfig: {
      maxTokens: 1024,
      temperature: 0.1,
    },
  }),
});
```

The `modelId` accepts standard Bedrock model IDs and cross-region inference profile IDs with region prefixes (e.g., `us.`, `eu.`, `global.`).

> **Instructions placeholders:** Instructions must contain placeholders appropriate for the evaluation level (e.g., `{context}`, `{available_tools}` for SESSION level). Evaluators using reference-input placeholders (e.g., `{expected_tool_trajectory}`, `{assertions}`) are only compatible with on-demand evaluation, not online evaluation. See the [custom evaluators documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/custom-evaluators.html) for allowed placeholders per level.

#### Code-Based Evaluator

Create a custom evaluator that uses a Lambda function for evaluation logic:

```typescript fixture=default
declare const evalFunction: lambda.IFunction;

const codeEvaluator = new agentcore.Evaluator(this, 'CodeEvaluator', {
  evaluatorName: 'custom_code_evaluator',
  level: agentcore.EvaluationLevel.TOOL_CALL,
  description: 'Evaluates tool call accuracy using custom logic',
  evaluatorConfig: agentcore.EvaluatorConfig.codeBased({
    lambdaFunction: evalFunction,
    timeout: cdk.Duration.seconds(30),
  }),
});
```

For code-based evaluators, the construct automatically grants the `bedrock-agentcore.amazonaws.com` service principal permission to invoke the Lambda function, scoped to the specific evaluator resource with `aws:SourceAccount` and `aws:SourceArn` conditions for confused deputy prevention.

#### Using Custom Evaluators with Online Evaluation

Custom evaluators are used in `OnlineEvaluationConfig` via `EvaluatorSelector.custom()`, alongside built-in evaluators:

```typescript fixture=default
declare const customEvaluator: agentcore.Evaluator;

const evaluation = new agentcore.OnlineEvaluationConfig(this, 'MixedEvaluation', {
  onlineEvaluationConfigName: 'mixed_evaluation',
  evaluators: [
    // Built-in evaluators
    agentcore.EvaluatorSelector.builtin(agentcore.BuiltinEvaluator.HELPFULNESS),
    agentcore.EvaluatorSelector.builtin(agentcore.BuiltinEvaluator.CORRECTNESS),
    // Custom evaluator
    agentcore.EvaluatorSelector.custom(customEvaluator),
  ],
  dataSource: agentcore.DataSourceConfig.fromCloudWatchLogs({
    logGroupNames: ['/aws/bedrock-agentcore/my-agent'],
    serviceNames: ['my-agent.default'],
  }),
});
```

### Data Source Configuration

Online evaluation supports two types of data sources:

**AgentCore Runtime Data Source (Recommended):**

For runtimes created within your CDK app, use `fromAgentRuntimeEndpoint()` which automatically derives the CloudWatch log group and service name:

```typescript fixture=default
const repository = new ecr.Repository(this, 'TestRepository', {
  repositoryName: 'test-agent-runtime',
});

const runtime = new agentcore.Runtime(this, 'MyRuntime', {
  runtimeName: 'my_agent',
  agentRuntimeArtifact: agentcore.AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0'),
});

// Using default endpoint (simplest)
const evaluation = new agentcore.OnlineEvaluationConfig(this, 'RuntimeEval', {
  onlineEvaluationConfigName: 'runtime_evaluation',
  evaluators: [
    agentcore.EvaluatorSelector.builtin(agentcore.BuiltinEvaluator.HELPFULNESS),
  ],
  dataSource: agentcore.DataSourceConfig.fromAgentRuntimeEndpoint(runtime),
});
```

You can also specify a specific endpoint:

```typescript fixture=default
declare const runtime: agentcore.Runtime;

// Using a specific endpoint construct
const prodEndpoint = runtime.addEndpoint('PROD');
const evaluation = new agentcore.OnlineEvaluationConfig(this, 'ProdEval', {
  onlineEvaluationConfigName: 'prod_evaluation',
  evaluators: [
    agentcore.EvaluatorSelector.builtin(agentcore.BuiltinEvaluator.CORRECTNESS),
  ],
  dataSource: agentcore.DataSourceConfig.fromAgentRuntimeEndpoint(runtime, prodEndpoint),
});

// Or using endpoint name as string
const stagingEval = new agentcore.OnlineEvaluationConfig(this, 'StagingEval', {
  onlineEvaluationConfigName: 'staging_evaluation',
  evaluators: [
    agentcore.EvaluatorSelector.builtin(agentcore.BuiltinEvaluator.CORRECTNESS),
  ],
  dataSource: agentcore.DataSourceConfig.fromAgentRuntimeEndpointName(runtime, 'STAGING'),
});
```

**CloudWatch Logs Data Source:**

For external agents or when you need to specify log groups directly:

```typescript fixture=default
const evaluation = new agentcore.OnlineEvaluationConfig(this, 'CloudWatchEval', {
  onlineEvaluationConfigName: 'cloudwatch_evaluation',
  evaluators: [
    agentcore.EvaluatorSelector.builtin(agentcore.BuiltinEvaluator.HELPFULNESS),
  ],
  dataSource: agentcore.DataSourceConfig.fromCloudWatchLogs({
    logGroupNames: [
      '/aws/bedrock-agentcore/agent1',
      '/aws/bedrock-agentcore/agent2',
    ],
    serviceNames: ['agent1.default'],
  }),
});
```

### Sampling and Filtering

Configure sampling percentage and filters to control which traces are evaluated:

```typescript fixture=default
const evaluation = new agentcore.OnlineEvaluationConfig(this, 'FilteredEval', {
  onlineEvaluationConfigName: 'filtered_evaluation',
  evaluators: [
    agentcore.EvaluatorSelector.builtin(agentcore.BuiltinEvaluator.HELPFULNESS),
  ],
  dataSource: agentcore.DataSourceConfig.fromCloudWatchLogs({
    logGroupNames: ['/aws/bedrock-agentcore/my-agent'],
    serviceNames: ['my-agent.default'],
  }),
  // Sample 25% of traces
  samplingPercentage: 25,
  // Only evaluate traces matching these filters
  filters: [
    {
      key: 'user.region',
      operator: agentcore.FilterOperator.EQUAL,
      value: agentcore.FilterValue.string('us-east-1'),
    },
    {
      key: 'session.duration',
      operator: agentcore.FilterOperator.GREATER_THAN,
      value: agentcore.FilterValue.number(60),
    },
  ],
  // Consider sessions complete after 30 minutes of inactivity
  sessionTimeout: cdk.Duration.minutes(30),
});
```

### Online Evaluation with Custom Execution Role

Provide a custom IAM role for the evaluation execution:

```typescript fixture=default
const executionRole = new iam.Role(this, 'EvaluationRole', {
  assumedBy: new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com'),
  description: 'Custom role for online evaluation',
});

// Add required permissions
executionRole.addToPolicy(new iam.PolicyStatement({
  actions: [
    'logs:DescribeLogGroups',
    'logs:GetQueryResults',
    'logs:StartQuery',
  ],
  resources: ['arn:aws:logs:*:*:log-group:/aws/bedrock-agentcore/*'],
}));

const evaluation = new agentcore.OnlineEvaluationConfig(this, 'CustomRoleEval', {
  onlineEvaluationConfigName: 'custom_role_evaluation',
  evaluators: [
    agentcore.EvaluatorSelector.builtin(agentcore.BuiltinEvaluator.HELPFULNESS),
  ],
  dataSource: agentcore.DataSourceConfig.fromCloudWatchLogs({
    logGroupNames: ['/aws/bedrock-agentcore/my-agent'],
    serviceNames: ['my-agent.default'],
  }),
  executionRole: executionRole,
});
```

### Online Evaluation IAM Permissions

Grant IAM permissions to manage or read evaluation configurations:

```typescript fixture=default
declare const evaluation: agentcore.OnlineEvaluationConfig;
declare const role: iam.IRole;

// Grant specific permissions
evaluation.grant(role,
  'bedrock-agentcore:GetOnlineEvaluationConfig',
  'bedrock-agentcore:UpdateOnlineEvaluationConfig',
);
```
