#!/usr/bin/env pwsh

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

#Requires -Version 6.0
#Requires -PSEdition Core
#Requires -Modules @{ModuleName='Az.Accounts'; ModuleVersion='1.6.4'}
#Requires -Modules @{ModuleName='Az.Resources'; ModuleVersion='1.8.0'}

[CmdletBinding(DefaultParameterSetName = 'Default')]
param (
    [Parameter(ParameterSetName = 'Default', Position = 0)]
    [string] $ServiceDirectory,

    [Parameter(ParameterSetName = 'Default')]
    [ValidatePattern('^[-a-zA-Z0-9\.\(\)_]{0,80}(?<=[a-zA-Z0-9\(\)])$')]
    [string] $BaseName,

    [Parameter(ParameterSetName = 'ResourceGroup')]
    [ValidatePattern('^[-\w\._\(\)]+$')]
    [string] $ResourceGroupName,

    [Parameter()]
    [ValidatePattern('^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$')]
    [string] $SubscriptionId,

    [Parameter()]
    [ValidateRange(1, [int]::MaxValue)]
    [int] $DeleteAfterHours = 48
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

# Support actions to invoke on exit.
$exitActions = @({
    if ($exitActions.Count -gt 1) {
        Write-Verbose 'Running registered exit actions'
    }
})

# Make sure $ResourceGroupName is set.
if (!$ResourceGroupName) {
    # Make sure $BaseName is set.
    if (!$BaseName) {
        $UserName = if ($env:USER) { $env:USER } else { "${env:USERNAME}" }
        # Remove spaces, etc. that may be in $UserName
        $UserName = $UserName -replace '\W'

        $BaseName = "$UserName$ServiceDirectory"
        Log "BaseName was not set. Using default base name '$BaseName'"
    }

    $ResourceGroupName = "rg-$BaseName"
}

# This script is intended for interactive users. Make sure they are logged in or fail.
$context = Get-AzContext
if (!$context) {
    throw "You must be already logged in to use this script. Run 'Connect-AzAccount' and try again."
}

# If no subscription was specified, try to select the Azure SDK Developer Playground subscription.
# Ignore errors to leave the automatically selected subscription.
if ($SubscriptionId) {
    $currentSubcriptionId = $context.Subscription.Id
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
    Log "Attempting to select subscription 'Azure SDK Developer Playground (faa080af-c1d8-40ad-9cce-e1a450ca5b57)'"
    $null = Select-AzSubscription -Subscription 'faa080af-c1d8-40ad-9cce-e1a450ca5b57' -ErrorAction Ignore

    # Update the context.
    $context = Get-AzContext

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

Log "Selected subscription '$subscriptionName'"

# try..finally will also trap Ctrl+C.
try {
    Log "Getting resource group '$ResourceGroupName'"

    $resourceGroup = Get-AzResourceGroup -Name $ResourceGroupName

    # Update DeleteAfter
    $deleteAfter = [DateTime]::UtcNow.AddHours($DeleteAfterHours).ToString('o')

    Log "Updating DeleteAfter to '$deleteAfter'"
    Write-Warning "Any clean-up scripts running against subscription '$SubscriptionId' may delete resource group '$ResourceGroupName' after $DeleteAfterHours hours."
    $resourceGroup.Tags['DeleteAfter'] = $deleteAfter

    Log "Updating resource group '$ResourceGroupName'"
    Retry {
        # Allow the resource group to write to output.
        Set-AzResourceGroup -Name $ResourceGroupName -Tag $resourceGroup.Tags
    }
} finally {
    $exitActions.Invoke()
}

<#
.SYNOPSIS
Updates a resource group previously deployed for a service directory.

.DESCRIPTION
Updates a resource group that was created using New-TestResources.ps1.
You can use this, for example, to update the `DeleteAfterHours` property
to keep an existing resource group deployed for a longer period of time.

.PARAMETER ServiceDirectory
A directory under 'sdk' in the repository root - optionally with subdirectories
specified - in which to discover ARM templates named 'test-resources.json'.
This can also be an absolute path or specify parent directories.

.PARAMETER BaseName
A name to use in the resource group and passed to the ARM template as 'baseName'.
This will update the resource group named 'rg-<baseName>'

.PARAMETER ResourceGroupName
The name of the resource group to update.

.PARAMETER SubscriptionId
Optional subscription ID to use when deleting resources when logging in as a
provisioner. You can also use Set-AzContext if not provisioning.

If you do not specify a SubscriptionId and are not logged in, one will be
automatically selected for you by the Connect-AzAccount cmdlet.

Once you are logged in (or were previously), the selected SubscriptionId
will be used for subsequent operations that are specific to a subscription.

.PARAMETER DeleteAfterHours
Positive integer number of hours from the current time to set the
'DeleteAfter' tag on the created resource group. The computed value is a
timestamp of the form "2020-03-04T09:07:04.3083910Z".

An optional cleanup process can delete resource groups whose "DeleteAfter"
timestamp is less than the current time.

.EXAMPLE
Update-TestResources.ps1 keyvault -DeleteAfterHours 24

Update the 'rg-${USERNAME}keyvault` resource group to be deleted after 24
hours from now if a clean-up script is running against the current subscription.

.EXAMPLE
Update-TestResources.ps1 -ResourceGroupName rg-userkeyvault -Subscription fa9c6912-f641-4226-806c-5139584b89ca

Update the 'rg-userkeyvault' resource group to be deleted after 48
hours from now if a clean-up script is running against the subscription 'fa9c6912-f641-4226-806c-5139584b89ca'.

#>
