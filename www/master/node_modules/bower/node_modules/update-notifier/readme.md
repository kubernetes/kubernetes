# update-notifier [![Build Status](https://travis-ci.org/yeoman/update-notifier.svg?branch=master)](https://travis-ci.org/yeoman/update-notifier)

> Update notifications for your CLI app

![](screenshot.png)

Inform users of your package of updates in a non-intrusive way.

#### Table of Contents

- [Examples](#examples)
- [How](#how)
- [API](#api)
- [About](#about)


## Examples

### Simple example

```js
var updateNotifier = require('update-notifier');
var pkg = require('./package.json');

updateNotifier({pkg: pkg}).notify();
```

### Comprehensive example

```js
var updateNotifier = require('update-notifier');
var pkg = require('./package.json');

// Checks for available update and returns an instance
var notifier = updateNotifier({pkg: pkg});

// Notify using the built-in convenience method
notifier.notify();

// `notifier.update` contains some useful info about the update
console.log(notifier.update);
/*
{
	latest: '1.0.1',
	current: '1.0.0',
	type: 'patch', // possible values: latest, major, minor, patch, prerelease, build
	name: 'pageres'
}
*/
```

### Example with settings and custom message

```js
var notifier = updateNotifier({
	pkg: pkg,
	updateCheckInterval: 1000 * 60 * 60 * 24 * 7 // 1 week
});

console.log('Update available: ' + notifier.update.latest);
```


## How

Whenever you initiate the update notifier and it's not within the interval threshold, it will asynchronously check with npm in the background for available updates, then persist the result. The next time the notifier is initiated the result will be loaded into the `.update` property. This prevents any impact on your package startup performance.
The check process is done in a unref'ed [child process](http://nodejs.org/api/child_process.html#child_process_child_process_spawn_command_args_options). This means that if you call `process.exit`, the check will still be performed in its own process.


## API

### updateNotifier(options)

Checks if there is an available update. Accepts settings defined below. Returns an object with update info if there is an available update, otherwise `undefined`.

### options

#### pkg

Type: `object`

##### name

*Required*  
Type: `string`

##### version

*Required*  
Type: `string`

#### updateCheckInterval

Type: `number`  
Default: `1000 * 60 * 60 * 24` (1 day)

How often to check for updates.

#### callback(error, update)

Type: `function`  

Passing a callback here will make it check for an update directly and report right away. Not recommended as you won't get the benefits explained in [`How`](#how).

`update` is equal to `notifier.update`


### updateNotifier.notify([options])

Convenience method to display a notification message *(see screenshot)*.

Only notifies if there is an update and the process is [TTY](http://nodejs.org/api/tty.html).

#### options.defer

Type: `boolean`  
Default: `true`

Defer showing the notication to after the process has exited.


### User settings

Users of your module have the ability to opt-out of the update notifier by changing the `optOut` property to `true` in `~/.config/configstore/update-notifier-[your-module-name].yml`. The path is available in `notifier.config.path`.

Users can also opt-out by [setting the environment variable](https://github.com/sindresorhus/guides/blob/master/set-environment-variables.md) `NO_UPDATE_NOTIFIER` with any value.

You could also let the user opt-out on a per run basis:

```js
if (process.argv.indexOf('--no-update-notifier') === -1) {
	// run updateNotifier()
}
```


## About

The idea for this module came from the desire to apply the browser update strategy to CLI tools, where everyone is always on the latest version. We first tried automatic updating, which we discovered wasn't popular. This is the second iteration of that idea, but limited to just update notifications.

There are a bunch projects using it:

- [Yeoman](http://yeoman.io) - modern workflows for modern webapps

- [Bower](http://bower.io) - a package manager for the web

- [Pageres](https://github.com/sindresorhus/pageres) - responsive website screenshots

- [Node GH](http://nodegh.io) - GitHub command line tool

- [Hoodie CLI](http://hood.ie) - Hoodie command line tool

- [Roots](http://roots.cx) - a toolkit for advanced front-end development

[And 100+ more...](https://www.npmjs.org/browse/depended/update-notifier)


## License

[BSD license](http://opensource.org/licenses/bsd-license.php) and copyright Google
