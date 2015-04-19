# Install and Uninstall Hooks

Bower provides 3 separate hooks that can be used to trigger other automated tools during Bower usage.  Importantly, these hooks are intended to allow external tools to help wire up the newly installed components into the parent project and other similar tasks.  These hooks are not intended to provide a post-installation build step for component authors.  As such, the configuration for these hooks is provided in the `.bowerrc` file in the parent project's directory.

## Configuring

In `.bowerrc` do:

```js
{
	"scripts": {
		"preinstall": "<your command here>",
		"postinstall": "<your command here>",
		"preuninstall": "<your command here>"
	}
}
```

The value of each script hook may contain a % character.  When your script is called, the % will be replaced with a space-separated list of components being installed or uninstalled.

Your script will also include an environment variable `BOWER_PID` containing the PID of the parent Bower process that triggered the script.  This can be used to verify that a `preinstall` and `postinstall` steps are part of the same Bower process.