# Live Test Resource Management

Live test runs require pre-existing resources in Azure. This set of PowerShell
commands automates creation and teardown of live test resources for Desktop and
CI scenarios.

* [New-TestResources.ps1](./New-TestResources.ps1.md) - Create new test resources
for the given service.
* [Remove-TestResources.ps1](./New-TestResources.ps1.md) - Deletes resources

## On the Desktop

Run `New-TestResources.ps1` on your desktop to create live test resources for a
given service (e.g. Storage, Key Vault, etc.). The command will output
environment variables that need to be set when running the live tests.

See examples for how to create the needed Service Principals and execute live
tests.

## In CI

The `New-TestResources.ps1` script is invoked on each test job to create an
isolated environment for live tests. Test resource isolation makes it easier to
parallelize test runs.

## Other

PowerShell markdown documentation created with
[PlatyPS](https://github.com/PowerShell/platyPS)