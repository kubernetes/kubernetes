# Plugins

This directory contains support code for building Gnostic plugins and associated examples.

Plugins are used to process API descriptions and can perform tasks like documentation and
code generation. Plugins can be written in any language that is supported by the Protocol
Buffer tools.

This directory contains several sample plugins and two support tools that make it easier
to test plugins by running them standalone.

* `gnostic-plugin-request` is a plugin that captures a plugin request and writes it in
.json and binary .pb form. When the optional -verbose flag is provided, this plugin logs
the request to stdout.
* `gnostic-process-plugin-response` is a standalone tool that reads a plugin response on
stdin and handles the response in the same way that gnostic does.

For example, this writes the plugin request to local files `plugin-request.json` and
`plugin-request.pb`.

`% gnostic myapi.json --plugin-request-out=.`

Then a plugin can be run standalone:

`% gnostic-go-generator --plugin < plugin-request.pb > plugin-response.pb`

Then you can use the following to process the plugin response:

`% gnostic-process-plugin-response -output=. < plugin-response.pb`