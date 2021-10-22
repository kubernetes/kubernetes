## AWS SDK for Go Private packages ##
`private` is a collection of packages used internally by the SDK, and is subject to have breaking changes. This package is not `internal` so that if you really need to use its functionality, and understand breaking changes will be made, you are able to.

These packages will be refactored in the future so that the API generator and model parsers are exposed cleanly on their own. Making it easier for you to generate your own code based on the API models.
