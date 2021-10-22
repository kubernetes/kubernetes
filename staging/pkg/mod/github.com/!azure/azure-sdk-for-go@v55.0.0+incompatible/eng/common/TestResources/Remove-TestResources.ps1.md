---
external help file: -help.xml
Module Name:
online version:
schema: 2.0.0
---

# Remove-TestResources.ps1

## SYNOPSIS
Deletes the resource group deployed for a service directory from Azure.

## SYNTAX

### Default (Default)
```
Remove-TestResources.ps1 [-BaseName <String>] [-SubscriptionId <String>] [-ServiceDirectory] <String>
 [-Environment <String>] [-Force] [-RemoveTestResourcesRemainingArguments <Object>] [-WhatIf] [-Confirm]
 [<CommonParameters>]
```

### Default+Provisioner
```
Remove-TestResources.ps1 -BaseName <String> -TenantId <String> [-SubscriptionId <String>]
 -ProvisionerApplicationId <String> -ProvisionerApplicationSecret <String> [[-ServiceDirectory] <String>]
 [-Environment <String>] [-Force] [-RemoveTestResourcesRemainingArguments <Object>] [-WhatIf] [-Confirm]
 [<CommonParameters>]
```

### ResourceGroup+Provisioner
```
Remove-TestResources.ps1 -ResourceGroupName <String> -TenantId <String> [-SubscriptionId <String>]
 -ProvisionerApplicationId <String> -ProvisionerApplicationSecret <String> [[-ServiceDirectory] <String>]
 [-Environment <String>] [-Force] [-RemoveTestResourcesRemainingArguments <Object>] [-WhatIf] [-Confirm]
 [<CommonParameters>]
```

### ResourceGroup
```
Remove-TestResources.ps1 -ResourceGroupName <String> [-SubscriptionId <String>] [[-ServiceDirectory] <String>]
 [-Environment <String>] [-Force] [-RemoveTestResourcesRemainingArguments <Object>] [-WhatIf] [-Confirm]
 [<CommonParameters>]
```

## DESCRIPTION
Removes a resource group and all its resources previously deployed using
New-TestResources.ps1.
If you are not currently logged into an account in the Az PowerShell module,
you will be asked to log in with Connect-AzAccount.
Alternatively, you (or a
build pipeline) can pass $ProvisionerApplicationId and
$ProvisionerApplicationSecret to authenticate a service principal with access to
create resources.

## EXAMPLES

### EXAMPLE 1
```
Remove-TestResources.ps1 keyvault -Force
Use the currently logged-in account to delete the resources created for Key Vault testing.
```

### EXAMPLE 2
```
Remove-TestResources.ps1 `
    -ResourceGroupName "${env:AZURE_RESOURCEGROUP_NAME}" `
    -TenantId '$(TenantId)' `
    -ProvisionerApplicationId '$(AppId)' `
    -ProvisionerApplicationSecret '$(AppSecret)' `
    -Force `
    -Verbose `
When run in the context of an Azure DevOps pipeline, this script removes the
resource group whose name is stored in the environment variable
AZURE_RESOURCEGROUP_NAME.
```

## PARAMETERS

### -BaseName
A name to use in the resource group and passed to the ARM template as 'baseName'.
This will delete the resource group named 'rg-\<baseName\>'

```yaml
Type: String
Parameter Sets: Default
Aliases:

Required: False
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

```yaml
Type: String
Parameter Sets: Default+Provisioner
Aliases:

Required: True
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -ResourceGroupName
The name of the resource group to delete.

```yaml
Type: String
Parameter Sets: ResourceGroup+Provisioner, ResourceGroup
Aliases:

Required: True
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -TenantId
The tenant ID of a service principal when a provisioner is specified.

```yaml
Type: String
Parameter Sets: Default+Provisioner, ResourceGroup+Provisioner
Aliases:

Required: True
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -SubscriptionId
Optional subscription ID to use when deleting resources when logging in as a
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
A service principal ID to provision test resources when a provisioner is specified.

```yaml
Type: String
Parameter Sets: Default+Provisioner, ResourceGroup+Provisioner
Aliases:

Required: True
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -ProvisionerApplicationSecret
A service principal secret (password) to provision test resources when a provisioner is specified.

```yaml
Type: String
Parameter Sets: Default+Provisioner, ResourceGroup+Provisioner
Aliases:

Required: True
Position: Named
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -ServiceDirectory
A directory under 'sdk' in the repository root - optionally with subdirectories
specified - in which to discover pre removal script named 'remove-test-resources-pre.json'.

```yaml
Type: String
Parameter Sets: Default
Aliases:

Required: True
Position: 1
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

```yaml
Type: String
Parameter Sets: Default+Provisioner, ResourceGroup+Provisioner, ResourceGroup
Aliases:

Required: False
Position: 1
Default value: None
Accept pipeline input: False
Accept wildcard characters: False
```

### -Environment
Name of the cloud environment.
The default is the Azure Public Cloud
('PublicCloud')

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

### -Force
Force removal of resource group without asking for user confirmation

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

### -RemoveTestResourcesRemainingArguments
Captures any arguments not declared here (no parameter errors)

```yaml
Type: Object
Parameter Sets: (All)
Aliases:

Required: False
Position: Named
Default value: None
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
