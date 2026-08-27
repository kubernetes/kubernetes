/******************************************************************************
 *                               BROWSER
 *****************************************************************************/
/**
 * Permissions for the Browser tool
 * See https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazonbedrockagentcore.html
 */
/******************************************************************************
   *                         Data Plane Permissions
   *****************************************************************************/
/**
 * Permissions to manage a specific browser session
 */
export const BROWSER_SESSION_PERMS = [
  'bedrock-agentcore:GetBrowserSession',
  'bedrock-agentcore:ListBrowserSessions',
  'bedrock-agentcore:StartBrowserSession',
  'bedrock-agentcore:StopBrowserSession',
];

/**
 * Permissions to connect to a browser live view or automation stream
 */
export const BROWSER_STREAM_PERMS = [
  'bedrock-agentcore:UpdateBrowserStream',
  'bedrock-agentcore:ConnectBrowserAutomationStream',
  'bedrock-agentcore:ConnectBrowserLiveViewStream',
];

/******************************************************************************
   *                         Control Plane Permissions
   *****************************************************************************/
/**
 * Grants control plane operations to manage the browser (CRUD)
 */
export const BROWSER_ADMIN_PERMS = [
  'bedrock-agentcore:CreateBrowser',
  'bedrock-agentcore:DeleteBrowser',
  'bedrock-agentcore:GetBrowser',
  'bedrock-agentcore:ListBrowsers',
];

/**
 * Permissions for reading browser information
 */
export const BROWSER_READ_PERMS = [
  'bedrock-agentcore:GetBrowser',
  'bedrock-agentcore:GetBrowserSession',
];

/**
 * Permissions for listing browser resources
 */
export const BROWSER_LIST_PERMS = [
  'bedrock-agentcore:ListBrowsers',
  'bedrock-agentcore:ListBrowserSessions',
];

/**
 * Permissions for using browser functionality
 */
export const BROWSER_USE_PERMS = [
  'bedrock-agentcore:StartBrowserSession',
  'bedrock-agentcore:UpdateBrowserStream',
  'bedrock-agentcore:StopBrowserSession',
];

/**
 * Least-privilege S3 permissions the browser execution role needs to write
 * session recordings. Recording is a write-only, multipart-upload workload, so
 * only these write actions are required (no read, list-bucket, or delete).
 * See https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/browser-resource-session-management.html
 */
export const BROWSER_RECORDING_S3_PERMS = [
  's3:PutObject',
  's3:ListMultipartUploadParts',
  's3:AbortMultipartUpload',
];

/******************************************************************************
 *                               CODE INTERPRETER
 *****************************************************************************/
/**
 * Permissions for the Code Interpreter tool
 * See https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazonbedrockagentcore.html
 */
/******************************************************************************
   *                         Data Plane Permissions
   *****************************************************************************/
/**
 * Permissions to manage a specific code interpreter session
 */
export const CODE_INTERPRETER_SESSION_PERMS = [
  'bedrock-agentcore:GetCodeInterpreterSession',
  'bedrock-agentcore:ListCodeInterpreterSessions',
  'bedrock-agentcore:StartCodeInterpreterSession',
  'bedrock-agentcore:StopCodeInterpreterSession',
];

/**
 * Permissions to invoke a code interpreter
 */
export const CODE_INTERPRETER_INVOKE_PERMS = ['bedrock-agentcore:InvokeCodeInterpreter'];

/******************************************************************************
   *                         Control Plane Permissions
   *****************************************************************************/
/**
 * Grants control plane operations to manage the code interpreter (CRUD)
 */
export const CODE_INTERPRETER_ADMIN_PERMS = [
  'bedrock-agentcore:CreateCodeInterpreter',
  'bedrock-agentcore:DeleteCodeInterpreter',
  'bedrock-agentcore:GetCodeInterpreter',
  'bedrock-agentcore:ListCodeInterpreters',
];

/**
 * Permissions for reading code interpreter information
 */
export const CODE_INTERPRETER_READ_PERMS = [
  'bedrock-agentcore:GetCodeInterpreter',
  'bedrock-agentcore:GetCodeInterpreterSession',
];

/**
 * Permissions for listing code interpreter resources
 */
export const CODE_INTERPRETER_LIST_PERMS = [
  'bedrock-agentcore:ListCodeInterpreters',
  'bedrock-agentcore:ListCodeInterpreterSessions',
];

/**
 * Permissions for using code interpreter functionality
 */
export const CODE_INTERPRETER_USE_PERMS = [
  'bedrock-agentcore:StartCodeInterpreterSession',
  'bedrock-agentcore:InvokeCodeInterpreter',
  'bedrock-agentcore:StopCodeInterpreterSession',
];
