Retrieve Credentials with Go Plugin
===

This example demonstrates how you can take advantage of Go 1.8's new Plugin
functionality to retrieve AWS credentials dynamically from a plugin compiled
separate from your application.

Usage
---

Example Plugin
---

You can find the plugin at `plugin/plugin.go` nested within this example. The plugin
demonstrates what symbol the SDK will use when lookup up the credential provider
and the type signature that needs to be implemented.

Compile the plugin with:

   go build -tags example -o myPlugin.so -buildmode=plugin plugin/plugin.go

JSON Credentials File
---

This example plugin will read the credentials from a JSON file pointed to by 
the `PLUGIN_CREDS_FILE` environment variable. The contents of the file are
the credentials, Key, Secret, and Token. The `Token` filed does not need to be
set if your credentials do not have one.

```json
{
	"Key":    "MyAWSCredAccessKeyID",
	"Secret": "MyAWSCredSecretKey",
	"Token":  "MyAWSCredToken"
}
```

Example Application
---

The `main.go` file in this folder demonstrates how you can configure the SDK to 
use a plugin to retrieve credentials with.

Compile and run application:

  go build -tags example -o myApp main.go

  PLUGIN_CREDS_FILE=pathToCreds.json ./myApp myPlugin.so myBucket myObjectKey

