---
external help file: -help.xml
Module Name:
online version:
schema: 2.0.0
---

# New-TestResources.ps1

## SYNOPSIS
Deploys live test resources defined for a service directory to Azure.

## SYNTAX

### Default (Default)
```
New-TestResources.ps1 [-BaseName <String>] [-ResourceGroupName <String>] [-ServiceDirectory] <String>
 [-TestApplicationId <String>] [-TestApplicationSecret <String>] [-TestApplicationOid <String>]
 [-SubscriptionId <String>] [-DeleteAfterHours <Int32>] [-Location <String>] [-Environment <String>]
 [-ArmTemplateParameters <Hashtable>] [-AdditionalParameters <Hashtable>] [-EnvironmentVariables <Hashtable>]
 [-CI] [-Force] [-OutFile] [-WhatIf] [-Confirm] [<CommonParameters>]
```

### Provisioner
```
New-TestResources.ps1 [-BaseName <String>] [-ResourceGroupName <String>] [-ServiceDirectory] <String>
 [-TestApplicationId <String>] [-TestApplicationSecret <String>] [-TestApplicationOid <String>]
 -TenantId <String> [-SubscriptionId <String>] -ProvisionerApplicationId <String>
 -ProvisionerApplicationSecret <String> [-DeleteAfterHours <Int32>] [-Location <String>]
 [-Environment <String>] [-ArmTemplateParameters <Hashtable>] [-AdditionalParameters <Hashtable>]
 [-EnvironmentVariables <Hashtable>] [-CI] [-Force] [-OutFile] [-WhatIf] [-Confirm] [<CommonParameters>]
```

## DESCRIPTION
Deploys live test resouces specified in test-resources.json files to a resource
group.

This script searches the directory specified in $ServiceDirectory recursively
for files named test-resources.json.
All found test-resources.json files will be
deployed to the test resource group.

If no test-resources.json files are located the script exits without making
changes to the Azure environment.

A service principal must first be created before this script is run and passed
to $TestApplicationId and $TestApplicationSecret.
Test resources will grant this
service principal access.

This script uses credentials already specified in Connect-AzAccount or those
specified in $ProvisionerApplicationId and $ProvisionerApplicationSecret.

## EXAMPLES

### EXAMPLE 1
```
Connect-AzAccount -Subscription 'REPLACE_WITH_SUBSCRIPTION_ID'
New-TestResources.ps1 keyvault
```

Run this in a desktop environment to create new AAD apps and Service Principals
that can be used to provision resources and run live tests.

Requires PowerShell 7 to use ConvertFrom-SecureString -AsPlainText or convert
the SecureString to plaintext by another means.

### EXAMPLE 2
```
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
```

Run this in an Azure DevOps CI (with approrpiate variables configured) before
executing live tests.
The script will output variables as secrets (to enable
log redaction).

## PARAMETERS

### -BaseName
A name to use in the resource group and passed to the ARM template as 'baseName'.
Limit $BaseName to enough characters to be under limit plus prefixes specified in
the ARM template.
See also https://docs.microsoft.com/azure/architecture/best-practices/resource-naming

Note: The value specified for this parameter will be overriden and generated
by New-TestResources.ps1 if $CI is specified.

```yaml
Type: String
Parameter Sets: (All)
Aliases:

Required: False
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -ResourceGroupName
Set this value to deploy directly to a Resource Group that has already been
created.

```yaml
Type: String
Parameter Sets: (All)
Aliases:

Required: False
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -ServiceDirectory
A directory under 'sdk' in the repository root - optionally with subdirectories
specified - in which to discover ARM templates named 'test-resources.json'.
This can also be an absolute path or specify parent directories.

```yaml
Type: String
Parameter Sets: (All)
Aliases:

Required: True
Position: 1
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -TestApplicationId
The AAD Application ID to authenticate the test runner against deployed
resources.
Passed to the ARM template as 'testApplicationId'.

This application is used by the test runner to execute tests against the
live test resources.

```yaml
Type: String
Parameter Sets: (All)
Aliases:

Required: False
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -TestApplicationSecret
Optional service principal secret (password) to authenticate the test runner
against deployed resources.
Passed to the ARM template as
'testApplicationSecret'.

This application is used by the test runner to execute tests against the
live test resources.

```yaml
Type: String
Parameter Sets: (All)
Aliases:

Required: False
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -TestApplicationOid
Service Principal Object ID of the AAD Test application.
This is used to assign
permissions to the AAD application so it can access tested features on the live
test resources (e.g.
Role Assignments on resources).
It is passed as to the ARM
template as 'testApplicationOid'

For more information on the relationship between AAD Applications and Service
Principals see: https://docs.microsoft.com/azure/active-directory/develop/app-objects-and-service-principals

```yaml
Type: String
Parameter Sets: (All)
Aliases:

Required: False
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -TenantId
The tenant ID of a service principal when a provisioner is specified.
The same
Tenant ID is used for Test Application and Provisioner Application.
This value
is passed to the ARM template as 'tenantId'.

```yaml
Type: String
Parameter Sets: Provisioner
Aliases:

Required: True
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -SubscriptionId
Optional subscription ID to use for new resources when logging in as a
provisioner.
You can also use Set-AzContext if not provisioning.

If you do not specify a SubscriptionId and are not logged in, one will be
automatically selected for you by the Connect-AzAccount cmdlet.

Once you are logged in (or were previously), the selected SubscriptionId
will be used for subsequent operations that are specific to a subscription.

```yaml
Type: String
Parameter Sets: (All)
Aliases:

Required: False
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -ProvisionerApplicationId
The AAD Application ID used to provision test resources when a provisioner is
specified.

If none is specified New-TestResources.ps1 uses the TestApplicationId.

This value is not passed to the ARM template.

```yaml
Type: String
Parameter Sets: Provisioner
Aliases:

Required: True
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -ProvisionerApplicationSecret
A service principal secret (password) used to provision test resources when a
provisioner is specified.

If none is specified New-TestResources.ps1 uses the TestApplicationSecret.

This value is not passed to the ARM template.

```yaml
Type: String
Parameter Sets: Provisioner
Aliases:

Required: True
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -DeleteAfterHours
Positive integer number of hours from the current time to set the
'DeleteAfter' tag on the created resource group.
The computed value is a
timestamp of the form "2020-03-04T09:07:04.3083910Z".

An optional cleanup process can delete resource groups whose "DeleteAfter"
timestamp is less than the current time.

This is used for CI automation.

```yaml
Type: Int32
Parameter Sets: (All)
Aliases:

Required: False
Position: Named
Default value: 48
Accept pipeline input: False
Accept wildcard characters: False
```

### -Location
Optional location where resources should be created.
If left empty, the default
is based on the cloud to which the template is being deployed:

* AzureCloud -\> 'westus2'
* AzureUSGovernment -\> 'usgovvirginia'
* AzureChinaCloud -\> 'chinaeast2'
* Dogfood -\> 'westus'

```yaml
Type: String
Parameter Sets: (All)
Aliases:

Required: False
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -Environment
Name of the cloud environment.
The default is the Azure Public Cloud
('AzureCloud')

```yaml
Type: String
Parameter Sets: (All)
Aliases:

Required: False
Position: Named
Default value: AzureCloud
Accept pipeline input: False
Accept wildcard characters: False
```

### -ArmTemplateParameters
Optional key-value pairs of parameters to pass to the ARM template(s).

```yaml
Type: Hashtable
Parameter Sets: (All)
Aliases:

Required: False
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -AdditionalParameters
Optional key-value pairs of parameters to pass to the ARM template(s) and pre-post scripts.

```yaml
Type: Hashtable
Parameter Sets: (All)
Aliases:

Required: False
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -EnvironmentVariables
Optional key-value pairs of parameters to set as environment variables to the shell.

```yaml
Type: Hashtable
Parameter Sets: (All)
Aliases:

Required: False
Position: Named
Default value: @{}
Accept pipeline input: False
Accept wildcard characters: False
```

### -CI
Indicates the script is run as part of a Continuous Integration / Continuous
Deployment (CI/CD) build (only Azure Pipelines is currently supported).

```yaml
Type: SwitchParameter
Parameter Sets: (All)
Aliases:

Required: False
Position: Named
Default value: ($null -ne $env:SYSTEM_TEAMPROJECTID)
Accept pipeline input: False
Accept wildcard characters: False
```

### -Force
Force creation of resources instead of being prompted.

```yaml
Type: SwitchParameter
Parameter Sets: (All)
Aliases:

Required: False
Position: Named
Default value: False
Accept pipeline input: False
Accept wildcard characters: False
```

### -OutFile
Save test environment settings into a test-resources.json.env file next to test-resources.json.
File is protected via DPAPI.
Supported only on windows.
The environment file would be scoped to the current repository directory.

```yaml
Type: SwitchParameter
Parameter Sets: (All)
Aliases:

Required: False
Position: Named
Default value: False
Accept pipeline input: False
Accept wildcard characters: False
```

### -WhatIf
Shows what would happen if the cmdlet runs.
The cmdlet is not run.

```yaml
Type: SwitchParameter
Parameter Sets: (All)
Aliases: wi

Required: False
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -Confirm
Prompts you for confirmation before running the cmdlet.

```yaml
Type: SwitchParameter
Parameter Sets: (All)
Aliases: cf

Required: False
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### CommonParameters
This cmdlet supports the common parameters: -Debug, -ErrorAction, -ErrorVariable, -InformationAction, -InformationVariable, -OutVariable, -OutBuffer, -PipelineVariable, -Verbose, -WarningAction, and -WarningVariable. For more information, see [about_CommonParameters](https://go.microsoft.com/fwlink/?LinkID=113216).

## INPUTS

## OUTPUTS

## NOTES

## RELATED LINKS
