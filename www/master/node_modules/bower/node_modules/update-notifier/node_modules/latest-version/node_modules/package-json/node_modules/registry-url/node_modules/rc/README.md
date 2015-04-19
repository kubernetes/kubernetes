# rc

The non-configurable configuration loader for lazy people.

## Usage

The only option is to pass rc the name of your app, and your default configuration.

```javascript
var conf = require('rc')(appname, {
  //defaults go here.
  port: 2468,

  //defaults which are objects will be merged, not replaced
  views: {
    engine: 'jade'
  }
});
```

`rc` will return your configuration options merged with the defaults you specify.
If you pass in a predefined defaults object, it will be mutated:

```javascript
var conf = {};
require('rc')(appname, conf);
```

If `rc` finds any config files for your app, the returned config object will have
a `configs` array containing their paths:

```javascript
var appCfg = require('rc')(appname, conf);
appCfg.configs[0] // /etc/appnamerc
appCfg.configs[1] // /home/dominictarr/.config/appname
appCfg.config // same as appCfg.configs[appCfg.configs.length - 1]
```

## Standards

Given your application name (`appname`), rc will look in all the obvious places for configuration.

  * command line arguments (parsed by minimist)
  * environment variables prefixed with `${appname}_`
    * or use "\_\_" to indicate nested properties <br/> _(e.g. `appname_foo__bar__baz` => `foo.bar.baz`)_
  * if you passed an option `--config file` then from that file
  * a local `.${appname}rc` or the first found looking in `./ ../ ../../ ../../../` etc.
  * `$HOME/.${appname}rc`
  * `$HOME/.${appname}/config`
  * `$HOME/.config/${appname}`
  * `$HOME/.config/${appname}/config`
  * `/etc/${appname}rc`
  * `/etc/${appname}/config`
  * the defaults object you passed in.

All configuration sources that were found will be flattened into one object,
so that sources **earlier** in this list override later ones.


## Configuration File Formats

Configuration files (e.g. `.appnamerc`) may be in either [json](http://json.org/example) or [ini](http://en.wikipedia.org/wiki/INI_file) format. The example configurations below are equivalent:


#### Formatted as `ini`

```
; You can include comments in `ini` format if you want.

dependsOn=0.10.0


; `rc` has built-in support for ini sections, see?

[commands]
  www     = ./commands/www
  console = ./commands/repl


; You can even do nested sections

[generators.options]
  engine  = ejs

[generators.modules]
  new     = generate-new
  engine  = generate-backend

```

#### Formatted as `json`

```json
{
  // You can even comment your JSON, if you want
  "dependsOn": "0.10.0",
  "commands": {
    "www": "./commands/www",
    "console": "./commands/repl"
  },
  "generators": {
    "options": {
      "engine": "ejs"
    },
    "modules": {
      "new": "generate-new",
      "backend": "generate-backend"
    }
  }
}
```

Comments are stripped from JSON config via [strip-json-comments](https://github.com/sindresorhus/strip-json-comments).

> Since ini, and env variables do not have a standard for types, your application needs be prepared for strings.



## Advanced Usage

#### Pass in your own `argv`

You may pass in your own `argv` as the third argument to `rc`.  This is in case you want to [use your own command-line opts parser](https://github.com/dominictarr/rc/pull/12).

```javascript
require('rc')(appname, defaults, customArgvParser);
```

## Pass in your own parser

If you have a special need to use a non-standard parser,
you can do so by passing in the parser as the 4th argument.
(leave the 3rd as null to get the default args parser)

```javascript
require('rc')(appname, defaults, null, parser);
```

This may also be used to force a more strict format,
such as strict, valid JSON only.

## Note on Performance

`rc` is running `fs.statSync`-- so make sure you don't use it in a hot code path (e.g. a request handler) 


## License

BSD / MIT / Apache2
