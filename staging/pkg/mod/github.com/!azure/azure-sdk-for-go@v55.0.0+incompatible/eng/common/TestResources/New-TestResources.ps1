#!/usr/bin/env pwsh

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

#Requires -Version 6.0
#Requires -PSEdition Core
#Requires -Modules @{ModuleName='Az.Accounts'; ModuleVersion='1.6.4'}
#Requires -Modules @{ModuleName='Az.Resources'; ModuleVersion='1.8.0'}

[CmdletBinding(DefaultParameterSetName = 'Default', SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param (
    # Limit $BaseName to enough characters to be under limit plus prefixes, and https://docs.microsoft.com/azure/architecture/best-practices/resource-naming
    [Parameter()]
    [ValidatePattern('^[-a-zA-Z0-9\.\(\)_]{0,80}(?<=[a-zA-Z0-9\(\)])$')]
    [string] $BaseName,

    [ValidatePattern('^[-\w\._\(\)]+$')]
    [string] $ResourceGroupName,

    [Parameter(Mandatory = $true, Position = 0)]
    [string] $ServiceDirectory,

    [Parameter()]
    [ValidatePattern('^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$')]
    [string] $TestApplicationId,

    [Parameter()]
    [string] $TestApplicationSecret,

    [Parameter()]
    [ValidatePattern('^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$')]
    [string] $TestApplicationOid,

    [Parameter(ParameterSetName = 'Provisioner', Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $TenantId,

    # Azure SDK Developer Playground subscription
    [Parameter()]
    [ValidatePattern('^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$')]
    [string] $SubscriptionId,

    [Parameter(ParameterSetName = 'Provisioner', Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$')]
    [string] $ProvisionerApplicationId,

    [Parameter(ParameterSetName = 'Provisioner', Mandatory = $true)]
    [string] $ProvisionerApplicationSecret,

    [Parameter()]
    [ValidateRange(1, [int]::MaxValue)]
    [int] $DeleteAfterHours = 48,

    [Parameter()]
    [string] $Location = '',

    [Parameter()]
    [ValidateSet('AzureCloud', 'AzureUSGovernment', 'AzureChinaCloud', 'Dogfood')]
    [string] $Environment = 'AzureCloud',

    [Parameter()]
    [hashtable] $ArmTemplateParameters,

    [Parameter()]
    [hashtable] $AdditionalParameters,

    [Parameter()]
    [ValidateNotNull()]
    [hashtable] $EnvironmentVariables = @{},

    [Parameter()]
    [switch] $CI = ($null -ne $env:SYSTEM_TEAMPROJECTID),

    [Parameter()]
    [switch] $Force,

    [Parameter()]
    [switch] $OutFile
)

# By default stop for any error.
if (!$PSBoundParameters.ContainsKey('ErrorAction')) {
    $ErrorActionPreference = 'Stop'
}

function Log($Message) {
    Write-Host ('{0} - {1}' -f [DateTime]::Now.ToLongTimeString(), $Message)
}

function Retry([scriptblock] $Action, [int] $Attempts = 5) {
    $attempt = 0
    $sleep = 5

    while ($attempt -lt $Attempts) {
        try {
            $attempt++
            return $Action.Invoke()
        } catch {
            if ($attempt -lt $Attempts) {
                $sleep *= 2

                Write-Warning "Attempt $attempt failed: $_. Trying again in $sleep seconds..."
                Start-Sleep -Seconds $sleep
            } else {
                Write-Error -ErrorRecord $_
            }
        }
    }
}

function MergeHashes([hashtable] $source, [psvariable] $dest) {
    foreach ($key in $source.Keys) {
        if ($dest.Value.ContainsKey($key) -and $dest.Value[$key] -ne $source[$key]) {
            Write-Warning ("Overwriting '$($dest.Name).$($key)' with value '$($dest.Value[$key])' " +
                          "to new value '$($source[$key])'")
        }
        $dest.Value[$key] = $source[$key]
    }
}

# Support actions to invoke on exit.
$exitActions = @({
    if ($exitActions.Count -gt 1) {
        Write-Verbose 'Running registered exit actions'
    }
})

New-Variable -Name 'initialContext' -Value (Get-AzContext) -Option Constant
if ($initialContext) {
    $exitActions += {
        Write-Verbose "Restoring initial context: $($initialContext.Account)"
        $null = $initialContext | Select-AzContext
    }
}

# try..finally will also trap Ctrl+C.
try {

    # Enumerate test resources to deploy. Fail if none found.
    $repositoryRoot = "$PSScriptRoot/../../.." | Resolve-Path
    $root = [System.IO.Path]::Combine($repositoryRoot, "sdk", $ServiceDirectory) | Resolve-Path
    $templateFileName = 'test-resources.json'
    $templateFiles = @()

    Write-Verbose "Checking for '$templateFileName' files under '$root'"
    Get-ChildItem -Path $root -Filter $templateFileName -Recurse | ForEach-Object {
        $templateFile = $_.FullName

        Write-Verbose "Found template '$templateFile'"
        $templateFiles += $templateFile
    }

    if (!$templateFiles) {
        Write-Warning -Message "No template files found under '$root'"
        exit
    }

    $UserName =  if ($env:USER) { $env:USER } else { "${env:USERNAME}" }
    # Remove spaces, etc. that may be in $UserName
    $UserName = $UserName -replace '\W'

    # Make sure $BaseName is set.
    if ($CI) {
        $BaseName = 't' + (New-Guid).ToString('n').Substring(0, 16)
        Log "Generated base name '$BaseName' for CI build"
    } elseif (!$BaseName) {
        $BaseName = "$UserName$ServiceDirectory"
        Log "BaseName was not set. Using default base name '$BaseName'"
    }

    # Make sure pre- and post-scripts are passed formerly required arguments.
    $PSBoundParameters['BaseName'] = $BaseName

    # Try detecting repos that support OutFile and defaulting to it
    if (!$CI -and !$PSBoundParameters.ContainsKey('OutFile') -and $IsWindows) {
        # TODO: find a better way to detect the language
        if (Test-Path "$repositoryRoot/eng/service.proj") {
            $OutFile = $true
            Log "Detected .NET repository. Defaulting OutFile to true. Test environment settings would be stored into the file so you don't need to set environment variables manually."
        }
    }

    # If no location is specified use safe default locations for the given
    # environment. If no matching environment is found $Location remains an empty
    # string.
    if (!$Location) {
        $Location = @{
            'AzureCloud' = 'westus2';
            'AzureUSGovernment' = 'usgovvirginia';
            'AzureChinaCloud' = 'chinaeast2';
            'Dogfood' = 'westus'
        }[$Environment]

        Write-Verbose "Location was not set. Using default location for environment: '$Location'"
    }

    if (!$CI) {

        # Make sure the user is logged in to create a service principal.
        $context = Get-AzContext;
        if (!$context) {
            Log 'User not logged in. Logging in now...'
            $context = (Connect-AzAccount).Context
        }

        $currentSubcriptionId = $context.Subscription.Id

        # If no subscription was specified, try to select the Azure SDK Developer Playground subscription.
        # Ignore errors to leave the automatically selected subscription.
        if ($SubscriptionId) {
            if ($currentSubcriptionId -ne $SubscriptionId) {
                Log "Selecting subscription '$SubscriptionId'"
                $null = Select-AzSubscription -Subscription $SubscriptionId

                $exitActions += {
                    Log "Selecting previous subscription '$currentSubcriptionId'"
                    $null = Select-AzSubscription -Subscription $currentSubcriptionId
                }

                # Update the context.
                $context = Get-AzContext
            }
        } else {
            if ($currentSubcriptionId -ne 'faa080af-c1d8-40ad-9cce-e1a450ca5b57') {
                Log "Attempting to select subscription 'Azure SDK Developer Playground (faa080af-c1d8-40ad-9cce-e1a450ca5b57)'"
                $null = Select-AzSubscription -Subscription 'faa080af-c1d8-40ad-9cce-e1a450ca5b57' -ErrorAction Ignore

                # Update the context.
                $context = Get-AzContext
            }

            $SubscriptionId = $context.Subscription.Id
            $PSBoundParameters['SubscriptionId'] = $SubscriptionId
        }

        # Use cache of well-known team subs without having to be authenticated.
        $wellKnownSubscriptions = @{
            'faa080af-c1d8-40ad-9cce-e1a450ca5b57' = 'Azure SDK Developer Playground'
            'a18897a6-7e44-457d-9260-f2854c0aca42' = 'Azure SDK Engineering System'
            '2cd617ea-1866-46b1-90e3-fffb087ebf9b' = 'Azure SDK Test Resources'
        }

        # Print which subscription is currently selected.
        $subscriptionName = $context.Subscription.Id
        if ($wellKnownSubscriptions.ContainsKey($subscriptionName)) {
            $subscriptionName = '{0} ({1})' -f $wellKnownSubscriptions[$subscriptionName], $subscriptionName
        }

        Log "Using subscription '$subscriptionName'"

        # Make sure the TenantId is also updated from the current context.
        # PSBoundParameters is not updated to avoid confusing parameter sets.
        if (!$TenantId) {
            $TenantId = $context.Subscription.TenantId
        }

        # If no test application ID is specified during an interactive session, create a new service principal.
        if (!$TestApplicationId) {

            # Cache the created service principal in this session for frequent reuse.
            $servicePrincipal = if ($AzureTestPrincipal -and (Get-AzADServicePrincipal -ApplicationId $AzureTestPrincipal.ApplicationId) -and $AzureTestSubscription -eq $SubscriptionId) {
                Log "TestApplicationId was not specified; loading cached service principal '$($AzureTestPrincipal.ApplicationId)'"
                $AzureTestPrincipal
            } else {
                Log "TestApplicationId was not specified; creating a new service principal in subscription '$SubscriptionId'"
                $global:AzureTestPrincipal = New-AzADServicePrincipal -Role Owner -Scope "/subscriptions/$SubscriptionId"
                $global:AzureTestSubscription = $SubscriptionId

                Log "Created service principal '$($AzureTestPrincipal.ApplicationId)'"
                $AzureTestPrincipal
            }

            $TestApplicationId = $servicePrincipal.ApplicationId
            $TestApplicationSecret = (ConvertFrom-SecureString $servicePrincipal.Secret -AsPlainText);

            # Make sure pre- and post-scripts are passed formerly required arguments.
            $PSBoundParameters['TestApplicationId'] = $TestApplicationId
            $PSBoundParameters['TestApplicationSecret'] = $TestApplicationSecret
        }

        if (!$ProvisionerApplicationId) {
            $ProvisionerApplicationId = $TestApplicationId
            $ProvisionerApplicationSecret = $TestApplicationSecret
            $TenantId = $context.Tenant.Id
        }
    }

    # Log in as and run pre- and post-scripts as the provisioner service principal.
    if ($ProvisionerApplicationId) {
        $null = Disable-AzContextAutosave -Scope Process

        Log "Logging into service principal '$ProvisionerApplicationId'."
        Write-Warning 'Logging into service principal may fail until the principal is fully propagated.'

        $provisionerSecret = ConvertTo-SecureString -String $ProvisionerApplicationSecret -AsPlainText -Force
        $provisionerCredential = [System.Management.Automation.PSCredential]::new($ProvisionerApplicationId, $provisionerSecret)

        # Use the given subscription ID if provided.
        $subscriptionArgs = if ($SubscriptionId) {
            @{Subscription = $SubscriptionId}
        } else {
            @{}
        }

        $provisionerAccount = Retry {
            Connect-AzAccount -Force:$Force -Tenant $TenantId -Credential $provisionerCredential -ServicePrincipal -Environment $Environment @subscriptionArgs
        }

        $exitActions += {
            Write-Verbose "Logging out of service principal '$($provisionerAccount.Context.Account)'"

            # Only attempt to disconnect if the -WhatIf flag was not set. Otherwise, this call is not necessary and will fail.
            if ($PSCmdlet.ShouldProcess($ProvisionerApplicationId)) {
                $null = Disconnect-AzAccount -AzureContext $provisionerAccount.Context
            }
        }
    }

    # Get test application OID from ID if not already provided.
    if ($TestApplicationId -and !$TestApplicationOid) {
        $testServicePrincipal = Retry {
            Get-AzADServicePrincipal -ApplicationId $TestApplicationId
        }

        if ($testServicePrincipal -and $testServicePrincipal.Id) {
            $script:TestApplicationOid = $testServicePrincipal.Id
        }
    }

    # Determine the Azure context that the script is running in.
    $context = Get-AzContext;

    # If the ServiceDirectory is an absolute path use the last directory name
    # (e.g. D:\foo\bar\ -> bar)
    $serviceName = if (Split-Path -IsAbsolute $ServiceDirectory) {
        Split-Path -Leaf $ServiceDirectory
    } else {
        $ServiceDirectory
    }

    $ResourceGroupName = if ($ResourceGroupName) {
        $ResourceGroupName
    } elseif ($CI) {
        # Format the resource group name based on resource group naming recommendations and limitations.
        "rg-{0}-$BaseName" -f ($serviceName -replace '[\\\/:]', '-').Substring(0, [Math]::Min($serviceName.Length, 90 - $BaseName.Length - 4)).Trim('-')
    } else {
        "rg-$BaseName"
    }

    $tags = @{
        Creator = $UserName
        ServiceDirectory = $ServiceDirectory
    }

    # Tag the resource group to be deleted after a certain number of hours.
    Write-Warning "Any clean-up scripts running against subscription '$SubscriptionId' may delete resource group '$ResourceGroupName' after $DeleteAfterHours hours."
    $deleteAfter = [DateTime]::UtcNow.AddHours($DeleteAfterHours).ToString('o')
    $tags['DeleteAfter'] = $deleteAfter

    if ($CI) {
        # Add tags for the current CI job.
        $tags += @{
            BuildId = "${env:BUILD_BUILDID}"
            BuildJob = "${env:AGENT_JOBNAME}"
            BuildNumber = "${env:BUILD_BUILDNUMBER}"
            BuildReason = "${env:BUILD_REASON}"
        }

        # Set the resource group name variable.
        Write-Host "Setting variable 'AZURE_RESOURCEGROUP_NAME': $ResourceGroupName"
        Write-Host "##vso[task.setvariable variable=AZURE_RESOURCEGROUP_NAME;]$ResourceGroupName"
        if ($EnvironmentVariables.ContainsKey('AZURE_RESOURCEGROUP_NAME') -and `
            $EnvironmentVariables['AZURE_RESOURCEGROUP_NAME'] -ne $ResourceGroupName)
        {
            Write-Warning ("Overwriting 'EnvironmentVariables.AZURE_RESOURCEGROUP_NAME' with value " +
                "'$($EnvironmentVariables['AZURE_RESOURCEGROUP_NAME'])' " + "to new value '$($ResourceGroupName)'")
        }
        $EnvironmentVariables['AZURE_RESOURCEGROUP_NAME'] = $ResourceGroupName
    }

    Log "Creating resource group '$ResourceGroupName' in location '$Location'"
    $resourceGroup = Retry {
        New-AzResourceGroup -Name "$ResourceGroupName" -Location $Location -Tag $tags -Force:$Force
    }

    if ($resourceGroup.ProvisioningState -eq 'Succeeded') {
        # New-AzResourceGroup would've written an error and stopped the pipeline by default anyway.
        Write-Verbose "Successfully created resource group '$($resourceGroup.ResourceGroupName)'"
    }
    elseif (!$resourceGroup) {
        if (!$PSCmdlet.ShouldProcess($resourceGroupName)) {
            # If the -WhatIf flag was passed, there will be no resource group created. Fake it.
            $resourceGroup = [PSCustomObject]@{
                ResourceGroupName = $resourceGroupName
                Location = $Location
            }
        } else {
            Write-Error "Resource group '$ResourceGroupName' already exists." -Category ResourceExists -RecommendedAction "Delete resource group '$ResourceGroupName', or overwrite it when redeploying."
        }
    }

    # Populate the template parameters and merge any additional specified.
    $templateParameters = @{
        baseName = $BaseName
        testApplicationId = $TestApplicationId
        testApplicationOid = "$TestApplicationOid"
    }

    if ($TenantId) {
        $templateParameters.Add('tenantId', $TenantId)
    }
    if ($TestApplicationSecret) {
        $templateParameters.Add('testApplicationSecret', $TestApplicationSecret)
    }

    MergeHashes $ArmTemplateParameters $(Get-Variable templateParameters)
    MergeHashes $AdditionalParameters $(Get-Variable templateParameters)

    # Include environment-specific parameters only if not already provided as part of the "ArmTemplateParameters"
    if (($context.Environment.StorageEndpointSuffix) -and (-not ($templateParameters.ContainsKey('storageEndpointSuffix')))) {
        $templateParameters.Add('storageEndpointSuffix', $context.Environment.StorageEndpointSuffix)
    }

    # Try to detect the shell based on the parent process name (e.g. launch via shebang).
    $shell, $shellExportFormat = if (($parentProcessName = (Get-Process -Id $PID).Parent.ProcessName) -and $parentProcessName -eq 'cmd') {
        'cmd', 'set {0}={1}'
    } elseif (@('bash', 'csh', 'tcsh', 'zsh') -contains $parentProcessName) {
        'shell', 'export {0}={1}'
    } else {
        'PowerShell', '${{env:{0}}} = ''{1}'''
    }

    # Deploy the templates
    foreach ($templateFile in $templateFiles) {
        # Deployment fails if we pass in more parameters than are defined.
        Write-Verbose "Removing unnecessary parameters from template '$templateFile'"
        $templateJson = Get-Content -LiteralPath $templateFile | ConvertFrom-Json
        $templateParameterNames = $templateJson.parameters.PSObject.Properties.Name

        $templateFileParameters = $templateParameters.Clone()
        foreach ($key in $templateParameters.Keys) {
            if ($templateParameterNames -notcontains $key) {
                Write-Verbose "Removing unnecessary parameter '$key'"
                $templateFileParameters.Remove($key)
            }
        }

        $preDeploymentScript = $templateFile | Split-Path | Join-Path -ChildPath 'test-resources-pre.ps1'
        if (Test-Path $preDeploymentScript) {
            Log "Invoking pre-deployment script '$preDeploymentScript'"
            &$preDeploymentScript -ResourceGroupName $ResourceGroupName @PSBoundParameters
        }

        Log "Deploying template '$templateFile' to resource group '$($resourceGroup.ResourceGroupName)'"
        $deployment = Retry {
            $lastDebugPreference = $DebugPreference
            try {
                if ($CI) {
                    $DebugPreference = 'Continue'
                }
                New-AzResourceGroupDeployment -Name $BaseName -ResourceGroupName $resourceGroup.ResourceGroupName -TemplateFile $templateFile -TemplateParameterObject $templateFileParameters -Force:$Force
            } catch {
                Write-Output @'
#####################################################
# For help debugging live test provisioning issues, #
# see http://aka.ms/azsdk/engsys/live-test-help,    #
#####################################################
'@
                throw
            } finally {
                $DebugPreference = $lastDebugPreference
            }
        }

        if ($deployment.ProvisioningState -eq 'Succeeded') {
            # New-AzResourceGroupDeployment would've written an error and stopped the pipeline by default anyway.
            Write-Verbose "Successfully deployed template '$templateFile' to resource group '$($resourceGroup.ResourceGroupName)'"
        }

        $serviceDirectoryPrefix = $serviceName.ToUpperInvariant() + "_"

        # Add default values
        $deploymentOutputs = @{
            "$($serviceDirectoryPrefix)CLIENT_ID" = $TestApplicationId;
            "$($serviceDirectoryPrefix)CLIENT_SECRET" = $TestApplicationSecret;
            "$($serviceDirectoryPrefix)TENANT_ID" = $context.Tenant.Id;
            "$($serviceDirectoryPrefix)SUBSCRIPTION_ID" =  $context.Subscription.Id;
            "$($serviceDirectoryPrefix)RESOURCE_GROUP" = $resourceGroup.ResourceGroupName;
            "$($serviceDirectoryPrefix)LOCATION" = $resourceGroup.Location;
            "$($serviceDirectoryPrefix)ENVIRONMENT" = $context.Environment.Name;
            "$($serviceDirectoryPrefix)AZURE_AUTHORITY_HOST" = $context.Environment.ActiveDirectoryAuthority;
            "$($serviceDirectoryPrefix)RESOURCE_MANAGER_URL" = $context.Environment.ResourceManagerUrl;
            "$($serviceDirectoryPrefix)SERVICE_MANAGEMENT_URL" = $context.Environment.ServiceManagementUrl;
            "$($serviceDirectoryPrefix)STORAGE_ENDPOINT_SUFFIX" = $context.Environment.StorageEndpointSuffix;
        }

        MergeHashes $EnvironmentVariables $(Get-Variable deploymentOutputs)

        foreach ($key in $deployment.Outputs.Keys) {
            $variable = $deployment.Outputs[$key]

            # Work around bug that makes the first few characters of environment variables be lowercase.
            $key = $key.ToUpperInvariant()

            if ($variable.Type -eq 'String' -or $variable.Type -eq 'SecureString') {
                $deploymentOutputs[$key] = $variable.Value
            }
        }

        if ($OutFile) {
            if (!$IsWindows) {
                Write-Host 'File option is supported only on Windows'
            }

            $outputFile = "$templateFile.env"

            $environmentText = $deploymentOutputs | ConvertTo-Json;
            $bytes = ([System.Text.Encoding]::UTF8).GetBytes($environmentText)
            $protectedBytes = [Security.Cryptography.ProtectedData]::Protect($bytes, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser)

            Set-Content $outputFile -Value $protectedBytes -AsByteStream -Force

            Write-Host "Test environment settings`n $environmentText`nstored into encrypted $outputFile"
        } else {

            if (!$CI) {
                # Write an extra new line to isolate the environment variables for easy reading.
                Log "Persist the following environment variables based on your detected shell ($shell):`n"
            }

            foreach ($key in $deploymentOutputs.Keys) {
                $value = $deploymentOutputs[$key]
                $EnvironmentVariables[$key] = $value

                if ($CI) {
                    # Treat all ARM template output variables as secrets since "SecureString" variables do not set values.
                    # In order to mask secrets but set environment variables for any given ARM template, we set variables twice as shown below.
                    Write-Host "Setting variable '$key': ***"
                    Write-Host "##vso[task.setvariable variable=_$key;issecret=true;]$($value)"
                    Write-Host "##vso[task.setvariable variable=$key;]$($value)"
                } else {
                    Write-Host ($shellExportFormat -f $key, $value)
                }
            }

            if ($key) {
                # Isolate the environment variables for easy reading.
                Write-Host "`n"
                $key = $null
            }
        }

        $postDeploymentScript = $templateFile | Split-Path | Join-Path -ChildPath 'test-resources-post.ps1'
        if (Test-Path $postDeploymentScript) {
            Log "Invoking post-deployment script '$postDeploymentScript'"
            &$postDeploymentScript -ResourceGroupName $ResourceGroupName -DeploymentOutputs $deploymentOutputs @PSBoundParameters
        }
    }

} finally {
    $exitActions.Invoke()
}

# Suppress output locally
if ($CI) {
    return $EnvironmentVariables
}

<#
.SYNOPSIS
Deploys live test resources defined for a service directory to Azure.

.DESCRIPTION
Deploys live test resouces specified in test-resources.json files to a resource
group.

This script searches the directory specified in $ServiceDirectory recursively
for files named test-resources.json. All found test-resources.json files will be
deployed to the test resource group.

If no test-resources.json files are located the script exits without making
changes to the Azure environment.

A service principal must first be created before this script is run and passed
to $TestApplicationId and $TestApplicationSecret. Test resources will grant this
service principal access.

This script uses credentials already specified in Connect-AzAccount or those
specified in $ProvisionerApplicationId and $ProvisionerApplicationSecret.

.PARAMETER BaseName
A name to use in the resource group and passed to the ARM template as 'baseName'.
Limit $BaseName to enough characters to be under limit plus prefixes specified in
the ARM template. See also https://docs.microsoft.com/azure/architecture/best-practices/resource-naming

Note: The value specified for this parameter will be overriden and generated
by New-TestResources.ps1 if $CI is specified.

.PARAMETER ResourceGroupName
Set this value to deploy directly to a Resource Group that has already been
created.

.PARAMETER ServiceDirectory
A directory under 'sdk' in the repository root - optionally with subdirectories
specified - in which to discover ARM templates named 'test-resources.json'.
This can also be an absolute path or specify parent directories.

.PARAMETER TestApplicationId
The AAD Application ID to authenticate the test runner against deployed
resources. Passed to the ARM template as 'testApplicationId'.

This application is used by the test runner to execute tests against the
live test resources.

.PARAMETER TestApplicationSecret
Optional service principal secret (password) to authenticate the test runner
against deployed resources. Passed to the ARM template as
'testApplicationSecret'.

This application is used by the test runner to execute tests against the
live test resources.

.PARAMETER TestApplicationOid
Service Principal Object ID of the AAD Test application. This is used to assign
permissions to the AAD application so it can access tested features on the live
test resources (e.g. Role Assignments on resources). It is passed as to the ARM
template as 'testApplicationOid'

For more information on the relationship between AAD Applications and Service
Principals see: https://docs.microsoft.com/azure/active-directory/develop/app-objects-and-service-principals

.PARAMETER TenantId
The tenant ID of a service principal when a provisioner is specified. The same
Tenant ID is used for Test Application and Provisioner Application. This value
is passed to the ARM template as 'tenantId'.

.PARAMETER SubscriptionId
Optional subscription ID to use for new resources when logging in as a
provisioner. You can also use Set-AzContext if not provisioning.

If you do not specify a SubscriptionId and are not logged in, one will be
automatically selected for you by the Connect-AzAccount cmdlet.

Once you are logged in (or were previously), the selected SubscriptionId
will be used for subsequent operations that are specific to a subscription.

.PARAMETER ProvisionerApplicationId
The AAD Application ID used to provision test resources when a provisioner is
specified.

If none is specified New-TestResources.ps1 uses the TestApplicationId.

This value is not passed to the ARM template.

.PARAMETER ProvisionerApplicationSecret
A service principal secret (password) used to provision test resources when a
provisioner is specified.

If none is specified New-TestResources.ps1 uses the TestApplicationSecret.

This value is not passed to the ARM template.

.PARAMETER DeleteAfterHours
Positive integer number of hours from the current time to set the
'DeleteAfter' tag on the created resource group. The computed value is a
timestamp of the form "2020-03-04T09:07:04.3083910Z".

An optional cleanup process can delete resource groups whose "DeleteAfter"
timestamp is less than the current time.

This is used for CI automation.

.PARAMETER Location
Optional location where resources should be created. If left empty, the default
is based on the cloud to which the template is being deployed:

* AzureCloud -> 'westus2'
* AzureUSGovernment -> 'usgovvirginia'
* AzureChinaCloud -> 'chinaeast2'
* Dogfood -> 'westus'

.PARAMETER Environment
Name of the cloud environment. The default is the Azure Public Cloud
('AzureCloud')

.PARAMETER AdditionalParameters
Optional key-value pairs of parameters to pass to the ARM template(s) and pre-post scripts.

.PARAMETER ArmTemplateParameters
Optional key-value pairs of parameters to pass to the ARM template(s).

.PARAMETER EnvironmentVariables
Optional key-value pairs of parameters to set as environment variables to the shell.

.PARAMETER CI
Indicates the script is run as part of a Continuous Integration / Continuous
Deployment (CI/CD) build (only Azure Pipelines is currently supported).

.PARAMETER Force
Force creation of resources instead of being prompted.

.PARAMETER OutFile
Save test environment settings into a test-resources.json.env file next to test-resources.json. File is protected via DPAPI. Supported only on windows.
The environment file would be scoped to the current repository directory.

.EXAMPLE
Connect-AzAccount -Subscription 'REPLACE_WITH_SUBSCRIPTION_ID'
New-TestResources.ps1 keyvault

Run this in a desktop environment to create new AAD apps and Service Principals
that can be used to provision resources and run live tests.

Requires PowerShell 7 to use ConvertFrom-SecureString -AsPlainText or convert
the SecureString to plaintext by another means.

.EXAMPLE
New-TestResources.ps1 `
    -BaseName 'Generated' `
    -ServiceDirectory '$(ServiceDirectory)' `
    -TenantId '$(TenantId)' `
    -ProvisionerApplicationId '$(ProvisionerId)' `
    -ProvisionerApplicationSecret '$(ProvisionerSecret)' `
    -TestApplicationId '$(TestAppId)' `
    -TestApplicationSecret '$(TestAppSecret)' `
    -DeleteAfterHours 24 `
    -CI `
    -Force `
    -Verbose

Run this in an Azure DevOps CI (with approrpiate variables configured) before
executing live tests. The script will output variables as secrets (to enable
log redaction).

#>
