//go:build go1.16
// +build go1.16

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/*
Package cloud implements a configuration API for applications deployed to sovereign or private Azure clouds.

Azure SDK client configuration defaults are appropriate for Azure Public Cloud (sometimes referred to as
"Azure Commercial" or simply "Microsoft Azure"). This package enables applications deployed to other
Azure Clouds to configure clients appropriately.

This package contains predefined configuration for well-known sovereign clouds such as Azure Government and
Azure China. Azure SDK clients accept this configuration via the Cloud field of azcore.ClientOptions. For
example, configuring a credential and ARM client for Azure Government:

	opts := azcore.ClientOptions{Cloud: cloud.AzureGovernment}
	cred, err := azidentity.NewDefaultAzureCredential(
		&azidentity.DefaultAzureCredentialOptions{ClientOptions: opts},
	)
	handle(err)

	client, err := armsubscription.NewClient(
		cred, &arm.ClientOptions{ClientOptions: opts},
	)
	handle(err)

Applications deployed to a private cloud such as Azure Stack create a Configuration object with
appropriate values:

	c := cloud.Configuration{
		ActiveDirectoryAuthorityHost: "https://...",
		Services: map[cloud.ServiceName]cloud.ServiceConfiguration{
			cloud.ResourceManager: {
				Audience: "...",
				Endpoint: "https://...",
			},
		},
	}
	opts := azcore.ClientOptions{Cloud: c}

	cred, err := azidentity.NewDefaultAzureCredential(
		&azidentity.DefaultAzureCredentialOptions{ClientOptions: opts},
	)
	handle(err)

	client, err := armsubscription.NewClient(
		cred, &arm.ClientOptions{ClientOptions: opts},
	)
	handle(err)
*/
package cloud
