# Live Test Resource Management

Running and recording live tests often requires first creating some resources
in Azure. Service directories that include a test-resources.json file require
running [New-TestResources.ps1][] to create these resources and output
environment variables you must set.

The following scripts can be used both in on your desktop for developer
scenarios as well as on hosted agents for continuous integration testing.

* [New-TestResources.ps1][] - Creates new test resources for a given service.
* [Remove-TestResources.ps1][] - Deletes previously created resources.

## Prerequisites

1. Install [PowerShell][] version 7.0 or newer.
2. Install the [Azure PowerShell][PowerShellAz].

## On the Desktop

To set up your Azure account to run live tests, you'll need to log into Azure,
and set up your resources defined in test-resources.json as shown in the following
example using Azure Key Vault. The script will create a service principal automatically,
or you may create a service principal you can save and reuse subsequently.

Note that `-Subscription` is an optional parameter but recommended if your account
is a member of multiple subscriptions. If you didn't specify it when logging in,
you should select your desired subscription using `Select-AzSubscription`. The
default can be saved using `Set-AzDefault` for future sessions.

```powershell
Connect-AzAccount -Subscription 'YOUR SUBSCRIPTION ID'
eng\common\TestResources\New-TestResources.ps1 keyvault
```

The `OutFile` switch will be set by default if you are running this for a .NET project on Windows. This will save test environment settings
into a test-resources.json.env file next to test-resources.json. The file is protected via DPAPI.
The environment file would be scoped to the current repository directory and avoids the need to
set environment variables or restart your IDE to recognize them.

Along with some log messages, this will output environment variables based on
your current shell like in the following example:

```powershell
${env:KEYVAULT_TENANT_ID} = '<<secret>>'
${env:KEYVAULT_CLIENT_ID} = '<<secret>>'
${env:KEYVAULT_CLIENT_SECRET} = '<<secret>>'
${env:KEYVAULT_SUBSCRIPTION_ID} = 'YOUR SUBSCRIPTION ID'
${env:KEYVAULT_RESOURCE_GROUP} = 'rg-myusername'
${env:KEYVAULT_LOCATION} = 'westus2'
${env:KEYVAULT_SKU} = 'premium'
${env:AZURE_KEYVAULT_URL} = '<<url>>'
```

For security reasons we do not set these environment variables automatically
for either the current process or persistently for future sessions. You must
do that yourself based on your current platform and shell.

If your current shell was detected properly, you should be able to copy and
paste the output directly in your terminal and add to your profile script.
For example, in PowerShell on Windows you can copy the output above and paste
it back into the terminal to set those environment variables for the current
process. To persist these variables for future terminal sessions or for
applications started outside the terminal, you could copy and paste the
following commands:

```powershell
setx KEYVAULT_TENANT_ID ${env:KEYVAULT_TENANT_ID}
setx KEYVAULT_CLIENT_ID ${env:KEYVAULT_CLIENT_ID}
setx KEYVAULT_CLIENT_SECRET ${env:KEYVAULT_CLIENT_SECRET}
setx KEYVAULT_SUBSCRIPTION_ID ${env:KEYVAULT_SUBSCRIPTION_ID}
setx KEYVAULT_RESOURCE_GROUP ${env:KEYVAULT_RESOURCE_GROUP}
setx KEYVAULT_LOCATION ${env:KEYVAULT_LOCATION}
setx KEYVAULT_SKU ${env:KEYVAULT_SKU}
setx AZURE_KEYVAULT_URL ${env:AZURE_KEYVAULT_URL}
```

### Cleaning up Resources

By default, resource groups are tagged with a `DeleteAfter` value and date according to the default or specified
value for the `-DeleteAfterHours` switch. You can use this tag in scheduled jobs to remove older resources based
on that date.

If you are not ready for the resources to be deleted, you can update the resource group by running [Update-TestResources.ps1][]:

```powershell
Update-TestResources.ps1 keyvault
```

This will extend the expiration time by the default value (e.g. 48 hours) from now.

Alternatively, after running or recording live tests, if you do not plan on further testing
you can immediately remove the test resources you created above by running [Remove-TestResources.ps1][]:

```powershell
Remove-TestResources.ps1 keyvault -Force
```

If you persisted environment variables, you should also remove those as well.

### Passing Additional Arguments

Some test-resources.json templates utilize the `AdditionalParameters` parameter to control additional resource configuration options. For example:

```powershell
New-TestResources.ps1 keyvault -AdditionalParameters @{enableHsm = $true}
```

## In CI

Test pipelines should include deploy-test-resources.yml and
remove-test-resources.yml like in the following examples:

```yml
- template: /eng/common/TestResources/deploy-test-resources.yml
  parameters:
    ServiceDirectory: '${{ parameters.ServiceDirectory }}'

# Run tests

- template: /eng/common/TestResources/remove-test-resources.yml
```

Be sure to link the **Secrets for Resource Provisioner** variable group
into the test pipeline for these scripts to work.

## Documentation

To regenerate documentation for scripts within this directory, you can install
[platyPS][] and run it like in the following example:

```powershell
Install-Module platyPS -Scope CurrentUser -Force
New-MarkdownHelp -Command .\New-TestResources.ps1 -OutputFolder . -Force
```

After the markdown files are generated, please make sure all "http" URIs use "https".

PowerShell markdown documentation created with [platyPS][].

  [New-TestResources.ps1]: https://aka.ms/azsdk/tools/New-TestResources
  [Update-TestResources.ps1]: https://aka.ms/azsdk/tools/Update-TestResources
  [Remove-TestResources.ps1]: https://aka.ms/azsdk/tools/Remove-TestResources
  [PowerShell]: https://github.com/PowerShell/PowerShell
  [PowerShellAz]: https://docs.microsoft.com/powershell/azure/install-az-ps
  [platyPS]: https://github.com/PowerShell/platyPS
