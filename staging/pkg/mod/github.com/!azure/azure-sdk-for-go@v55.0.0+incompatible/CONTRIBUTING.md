# Contributing

Most packages under the `services` directory in the SDK are generated from [Azure API specs][azure_rest_specs]
using [Azure/autorest.go][] and [Azure/autorest][]. These generated packages depend on the HTTP client implemented at [Azure/go-autorest][]. Therefore when contributing, please make sure you do not change anything under the `services` directory.

[azure_rest_specs]: https://github.com/Azure/azure-rest-api-specs
[azure/autorest]: https://github.com/Azure/autorest
[azure/autorest.go]: https://github.com/Azure/autorest.go
[azure/go-autorest]: https://github.com/Azure/go-autorest

For bugs or feature requests you can submit them using the [Github issues page][issues] and filling the appropriate template.

Also please see these [guidelines][] about contributing to Azure projects.

This project follows the [Microsoft Open Source Code of Conduct][coc]. For more information see the [Code of Conduct FAQ][cocfaq]. Contact [opencode@microsoft.com][cocmail] with questions and comments.

[guidelines]: https://opensource.microsoft.com/collaborate/
[coc]: https://opensource.microsoft.com/codeofconduct/
[cocfaq]: https://opensource.microsoft.com/codeofconduct/faq/
[cocmail]: mailto:opencode@microsoft.com
[issues]: https://github.com/Azure/Azure-sdk-for-go/issues
