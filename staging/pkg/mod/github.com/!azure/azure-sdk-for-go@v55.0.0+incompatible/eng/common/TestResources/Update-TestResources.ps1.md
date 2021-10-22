---
external help file: -help.xml
Module Name:
online version:
schema: 2.0.0
---

# Update-TestResources.ps1

## SYNOPSIS
Updates a resource group previously deployed for a service directory.

## SYNTAX

### Default (Default)
```
Update-TestResources.ps1 [-ServiceDirectory] <String> [-BaseName <String>] [-SubscriptionId <String>]
 [-DeleteAfterHours <Int32>] [<CommonParameters>]
```

### ResourceGroup
```
Update-TestResources.ps1 [-ResourceGroupName <String>] [-SubscriptionId <String>] [-DeleteAfterHours <Int32>]
 [<CommonParameters>]
```

## DESCRIPTION
Updates a resource group that was created using New-TestResources.ps1.
You can use this, for example, to update the \`DeleteAfterHours\` property
to keep an existing resource group deployed for a longer period of time.

## EXAMPLES

### EXAMPLE 1
```
Update-TestResources.ps1 keyvault -DeleteAfterHours 24
```

Update the 'rg-${USERNAME}keyvault\` resource group to be deleted after 24
hours from now if a clean-up script is running against the current subscription.

### EXAMPLE 2
```
Update-TestResources.ps1 -ResourceGroupName rg-userkeyvault -Subscription fa9c6912-f641-4226-806c-5139584b89ca
```

Update the 'rg-userkeyvault' resource group to be deleted after 48
hours from now if a clean-up script is running against the subscription 'fa9c6912-f641-4226-806c-5139584b89ca'.

## PARAMETERS

### -ServiceDirectory
A directory under 'sdk' in the repository root - optionally with subdirectories
specified - in which to discover ARM templates named 'test-resources.json'.
This can also be an absolute path or specify parent directories.

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

### -BaseName
A name to use in the resource group and passed to the ARM template as 'baseName'.
This will update the resource group named 'rg-\<baseName\>'

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

### -ResourceGroupName
The name of the resource group to update.

```yaml
Type: String
Parameter Sets: ResourceGroup
Aliases:

Required: False
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

### -DeleteAfterHours
Positive integer number of hours from the current time to set the
'DeleteAfter' tag on the created resource group.
The computed value is a
timestamp of the form "2020-03-04T09:07:04.3083910Z".

An optional cleanup process can delete resource groups whose "DeleteAfter"
timestamp is less than the current time.

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

### CommonParameters
This cmdlet supports the common parameters: -Debug, -ErrorAction, -ErrorVariable, -InformationAction, -InformationVariable, -OutVariable, -OutBuffer, -PipelineVariable, -Verbose, -WarningAction, and -WarningVariable. For more information, see [about_CommonParameters](https://go.microsoft.com/fwlink/?LinkID=113216).

## INPUTS

## OUTPUTS

## NOTES

## RELATED LINKS
