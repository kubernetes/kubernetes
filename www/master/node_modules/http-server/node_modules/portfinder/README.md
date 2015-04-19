# node-portfinder

## Installation

### Installing npm (node package manager)
``` bash
  curl http://npmjs.org/install.sh | sh
```

### Installing node-portfinder
``` bash
  $ [sudo] npm install portfinder
```

## Usage
The `portfinder` module has a simple interface:

``` js
  var portfinder = require('portfinder');
  
  portfinder.getPort(function (err, port) {
    //
    // `port` is guarenteed to be a free port 
    // in this scope.
    //
  });
```

By default `portfinder` will start searching from `8000`. To change this simply set `portfinder.basePort`.

## Run Tests
``` bash
  $ npm test
```

#### Author: [Charlie Robbins][0]
[0]: http://nodejitsu.com