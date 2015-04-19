# endpoint-parser [![Build Status](https://secure.travis-ci.org/bower/endpoint-parser.png?branch=master)](http://travis-ci.org/bower/endpoint-parser)

Little module that helps with endpoints parsing.


## API

### .decompose(endpoint)

Decomposes a endpoint into `name`, `source` and `target`.

```js
var endpointParser = require('bower-endpoint-parser');

endpointParser.decompose('jquery#~2.0.0');
// { name: '', source: 'jquery', target: '~2.0.0' }

endpointParser.decompose('backbone=backbone-amd#~1.0.0');
// { name: 'backbone', source: 'backbone-amd', target: '~1.0.0' }

endpointParser.decompose('http://twitter.github.io/bootstrap/assets/bootstrap.zip');
// { name: '', source: 'http://twitter.github.io/bootstrap/assets/bootstrap', target: '*' }

endpointParser.decompose('bootstrap=http://twitter.github.io/bootstrap/assets/bootstrap.zip');
// { name: 'bootstrap', source: 'http://twitter.github.io/bootstrap/assets/bootstrap', target: '*' }
```

### .compose(decEndpoint)

Inverse of `decompose()`.   
Takes a decomposed endpoint and composes it back into a string.

```js
var endpointParser = require('bower-endpoint-parser');

endpointParser.compose({ name: '', source: 'jquery', target: '~2.0.0' });
// jquery#~2.0.0

endpointParser.compose({ name: 'backbone', source: 'backbone-amd', target: '~1.0.0' });
// backbone=backbone-amd#~1.0.0

endpointParser.compose({ name: '', source: 'http://twitter.github.io/bootstrap/assets/bootstrap', target: '*' });
// http://twitter.github.io/bootstrap/assets/bootstrap.zip

endpointParser.compose({ name: 'bootstrap', source: 'http://twitter.github.io/bootstrap/assets/bootstrap', target: '*' });
// bootstrap=http://twitter.github.io/bootstrap/assets/bootstrap.zip
```

### .json2decomposed(key, value)

Similar to `decompose()` but specially designed to be used when parsing `bower.json` dependencies.
For instance, in a `bower.json` like this:

```js
{
    "name": "foo",
    "version": "0.1.0",
    "dependencies": {
        "jquery": "~1.9.1",
        "backbone": "backbone-amd#~1.0.0",
        "bootstrap": "http://twitter.github.io/bootstrap/assets/bootstrap"
    }
}
```

You would call `json2decomposed` like so:

```js
endpointParser.json2decomposed('jquery', '~1.9.1');
// { name: 'jquery', source: 'jquery', target: '~1.9.1' }

endpointParser.json2decomposed('backbone', 'backbone-amd#~1.0.0');
// { name: 'backbone', source: 'backbone-amd', target: '~1.0.0' }

endpointParser.json2decomposed('bootstrap', 'http://twitter.github.io/bootstrap/assets/bootstrap');
// { name: 'bootstrap', source: 'http://twitter.github.io/bootstrap/assets/bootstrap', target: '*' }
```

### .decomposed2json(decEndpoint)

Inverse of `json2decomposed()`.   
Takes a decomposed endpoint and composes it to be saved to `bower.json`.

```js
var endpointParser = require('bower-endpoint-parser');

endpointParser.decomposed2json({ name: 'jquery', source: 'jquery', target: '~2.0.0' });
// { jquery: '~2.0.0' }

endpointParser.decomposed2json({ name: 'backbone', source: 'backbone-amd', target: '~1.0.0' });
// { backbone: 'backbone-amd#~2.0.0' }

endpointParser.decomposed2json({ name: 'bootstrap', source: 'http://twitter.github.io/bootstrap/assets/bootstrap', target: '*' });
// { bootstrap: 'http://twitter.github.io/bootstrap/assets/bootstrap' }
```

This function throws an exception if the `name` from the decomposed endpoint is empty.


## License

Released under the [MIT License](http://www.opensource.org/licenses/mit-license.php).
