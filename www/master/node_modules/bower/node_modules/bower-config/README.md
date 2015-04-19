# bower-config [![Build Status](https://secure.travis-ci.org/bower/config.png?branch=master)](http://travis-ci.org/bower/config)

> The Bower config (`.bowerrc`) reader and writer.

The config spec can be read [here](https://docs.google.com/document/d/1APq7oA9tNao1UYWyOm8dKqlRP2blVkROYLZ2fLIjtWc/).


## Install

```sh
$ npm install --save bower-config
```


## Usage

#### .load()

Loads the bower configuration from the configuration files.


#### .get(key) - NOT YET IMPLEMENTED

Returns a configuration value by `key`.   
Keys with dots are supported to access deep values.


#### .set(key, value) - NOT YET IMPLEMENTED

Sets a configuration value for `key`.   
Keys with dots are supported to set deep values.


#### .del(key) - NOT YET IMPLEMENTED

Removes configuration named `key`.   
Keys with dots are supported to delete deep keys.


#### .save(where, callback) - NOT YET IMPLEMENTED

Saves changes to `where`.   
The `where` argument can be a path to a configuration file or:

- `local` to save it in the configured current working directory (defaulting to `process.cwd`)
- `user` to save it in the configuration file located in the home directory


#### .toObject()

Returns a deep copy of the underlying configuration object.   
The returned configuration is normalised.   
The object keys will be camelCase.


#### #create(cwd)

Obtains a instance where `cwd` is the current working directory (defaults to `process.cwd`);

```js
var config = require('bower-config').create();
// You can also specify a working directory
var config2 = require('bower-config').create('./some/path');
```


#### #read(cwd)

Alias for:

```js
var configObject = (new Config(cwd)).load().toJson();
```


#### #normalise(config)

Returns a new normalised config object based on `config`.   
Object keys will be converted to camelCase.


## License

Released under the [MIT License](http://www.opensource.org/licenses/mit-license.php).
