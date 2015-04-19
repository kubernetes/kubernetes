# configstore [![Build Status](https://secure.travis-ci.org/yeoman/configstore.svg?branch=master)](http://travis-ci.org/yeoman/configstore)

> Easily load and persist config without having to think about where and how.

Config is stored in a YAML file to make it simple for users to edit the config directly themselves. The file is located in `$XDG_CONFIG_HOME` or `~/.config`. Eg: `~/.config/configstore/id-of-your-choosing.yml`


## Usage

```js
var Configstore = require('configstore');
var packageName = require('./package').name;

// Init a Configstore instance with an unique ID eg. package name
// and optionally some default values
var conf = new Configstore(packageName, { foo: 'bar' });

conf.set('awesome', true);
console.log(conf.get('awesome'));  // true
console.log(conf.get('foo'));      // bar

conf.del('awesome');
console.log(conf.get('awesome'));  // undefined
```


## Documentation

### Methods

#### .set(key, val)

Set an item

#### .get(key)

Get an item

#### .del(key)

Delete an item

### Properties

#### .all

Get all items as an object or replace the current config with an object:

```js
conf.all = {
	hello: 'world'
};
```

#### .size

Get the item count

#### .path

Get the path to the config file. Can be used to show the user where the config file is located or even better open it for them.


## License

[BSD license](http://opensource.org/licenses/bsd-license.php)  
Copyright Google
