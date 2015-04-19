#config-chain

USE THIS MODULE TO LOAD ALL YOUR CONFIGURATIONS

``` js

  //npm install config-chain

  var cc = require('config-chain')
    , opts = require('optimist').argv //ALWAYS USE OPTIMIST FOR COMMAND LINE OPTIONS.
    , env = opts.env || process.env.YOUR_APP_ENV || 'dev' //SET YOUR ENV LIKE THIS.

  // EACH ARG TO CONFIGURATOR IS LOADED INTO CONFIGURATION CHAIN
  // EARLIER ITEMS OVERIDE LATER ITEMS
  // PUTS COMMAND LINE OPTS FIRST, AND DEFAULTS LAST!

  //strings are interpereted as filenames.
  //will be loaded synchronously

  var conf =
  cc(
    //OVERRIDE SETTINGS WITH COMMAND LINE OPTS
    opts,

    //ENV VARS IF PREFIXED WITH 'myApp_'

    cc.env('myApp_'), //myApp_foo = 'like this'

    //FILE NAMED BY ENV
    path.join(__dirname,  'config.' + env + '.json'),

    //IF `env` is PRODUCTION
    env === 'prod'
      ? path.join(__dirname, 'special.json') //load a special file
      : null //NULL IS IGNORED!

    //SUBDIR FOR ENV CONFIG
    path.join(__dirname,  'config', env, 'config.json'),

    //SEARCH PARENT DIRECTORIES FROM CURRENT DIR FOR FILE
    cc.find('config.json'),

    //PUT DEFAULTS LAST
    {
      host: 'localhost'
      port: 8000
    })

  var host = conf.get('host')

  // or

  var host = conf.store.host

```

FINALLY, EASY FLEXIBLE CONFIGURATIONS!

##see also: [proto-list](https://github.com/isaacs/proto-list/)

WHATS THAT YOU SAY?

YOU WANT A "CLASS" SO THAT YOU CAN DO CRAYCRAY JQUERY CRAPS?

EXTEND WITH YOUR OWN FUNCTIONALTY!?

## CONFIGCHAIN LIVES TO SERVE ONLY YOU!

```javascript
var cc = require('config-chain')

// all the stuff you did before
var config = cc({
      some: 'object'
    },
    cc.find('config.json'),
    cc.env('myApp_')
  )
  // CONFIGS AS A SERVICE, aka "CaaS", aka EVERY DEVOPS DREAM OMG!
  .addUrl('http://configurator:1234/my-configs')
  // ASYNC FTW!
  .addFile('/path/to/file.json')

  // OBJECTS ARE OK TOO, they're SYNC but they still ORDER RIGHT
  // BECAUSE PROMISES ARE USED BUT NO, NOT *THOSE* PROMISES, JUST
  // ACTUAL PROMISES LIKE YOU MAKE TO YOUR MOM, KEPT OUT OF LOVE
  .add({ another: 'object' })

  // DIE A THOUSAND DEATHS IF THIS EVER HAPPENS!!
  .on('error', function (er) {
    // IF ONLY THERE WAS SOMETHIGN HARDER THAN THROW
    // MY SORROW COULD BE ADEQUATELY EXPRESSED.  /o\
    throw er
  })

  // THROW A PARTY IN YOUR FACE WHEN ITS ALL LOADED!!
  .on('load', function (config) {
    console.awesome('HOLY SHIT!')
  })
```

# BORING API DOCS

## cc(...args)

MAKE A CHAIN AND ADD ALL THE ARGS.

If the arg is a STRING, then it shall be a JSON FILENAME.

SYNC I/O!

RETURN THE CHAIN!

## cc.json(...args)

Join the args INTO A JSON FILENAME!

SYNC I/O!

## cc.find(relativePath)

SEEK the RELATIVE PATH by climbing the TREE OF DIRECTORIES.

RETURN THE FOUND PATH!

SYNC I/O!

## cc.parse(content, file, type)

Parse the content string, and guess the type from either the
specified type or the filename.

RETURN THE RESULTING OBJECT!

NO I/O!

## cc.env(prefix, env=process.env)

Get all the keys on the provided env object (or process.env) which are
prefixed by the specified prefix, and put the values on a new object.

RETURN THE RESULTING OBJECT!

NO I/O!

## cc.ConfigChain()

The ConfigChain class for CRAY CRAY JQUERY STYLE METHOD CHAINING!

One of these is returned by the main exported function, as well.

It inherits (prototypically) from
[ProtoList](https://github.com/isaacs/proto-list/), and also inherits
(parasitically) from
[EventEmitter](http://nodejs.org/api/events.html#events_class_events_eventemitter)

It has all the methods from both, and except where noted, they are
unchanged.

### LET IT BE KNOWN THAT chain IS AN INSTANCE OF ConfigChain.

## chain.sources

A list of all the places where it got stuff.  The keys are the names
passed to addFile or addUrl etc, and the value is an object with some
info about the data source.

## chain.addFile(filename, type, [name=filename])

Filename is the name of the file.  Name is an arbitrary string to be
used later if you desire.  Type is either 'ini' or 'json', and will
try to guess intelligently if omitted.

Loaded files can be saved later.

## chain.addUrl(url, type, [name=url])

Same as the filename thing, but with a url.

Can't be saved later.

## chain.addEnv(prefix, env, [name='env'])

Add all the keys from the env object that start with the prefix.

## chain.addString(data, file, type, [name])

Parse the string and add it to the set.  (Mainly used internally.)

## chain.add(object, [name])

Add the object to the set.

## chain.root {Object}

The root from which all the other config objects in the set descend
prototypically.

Put your defaults here.

## chain.set(key, value, name)

Set the key to the value on the named config object.  If name is
unset, then set it on the first config object in the set.  (That is,
the one with the highest priority, which was added first.)

## chain.get(key, [name])

Get the key from the named config object explicitly, or from the
resolved configs if not specified.

## chain.save(name, type)

Write the named config object back to its origin.

Currently only supported for env and file config types.

For files, encode the data according to the type.

## chain.on('save', function () {})

When one or more files are saved, emits `save` event when they're all
saved.

## chain.on('load', function (chain) {})

When the config chain has loaded all the specified files and urls and
such, the 'load' event fires.
