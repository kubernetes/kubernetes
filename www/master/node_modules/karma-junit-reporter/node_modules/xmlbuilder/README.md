# xmlbuilder-js

An XMLBuilder for [node.js](http://nodejs.org/) similar to 
[java-xmlbuilder](http://code.google.com/p/java-xmlbuilder/).

[![Build Status](https://secure.travis-ci.org/oozcitak/xmlbuilder-js.png)](http://travis-ci.org/oozcitak/xmlbuilder-js)

### Installation:

``` sh
npm install xmlbuilder
```

### Important:

I had to break compatibility while adding multiple instances in 0.1.3. 
As a result, version from v0.1.3 are **not** compatible with previous versions.

### Usage:

``` js
var builder = require('xmlbuilder');
var xml = builder.create('root')
  .ele('xmlbuilder', {'for': 'node-js'})
    .ele('repo', {'type': 'git'}, 'git://github.com/oozcitak/xmlbuilder-js.git')
  .end({ pretty: true});
    
console.log(xml);
```

will result in:

``` xml
<?xml version="1.0"?>
<root>
  <xmlbuilder for="node-js">
    <repo type="git">git://github.com/oozcitak/xmlbuilder-js.git</repo>
  </xmlbuilder>
</root>
```

If you need to do some processing:

``` js
var root = builder.create('squares');
root.com('f(x) = x^2');
for(var i = 1; i <= 5; i++)
{
  var item = root.ele('data');
  item.att('x', i);
  item.att('y', i * i);
}
```

This will result in:

``` xml
<?xml version="1.0"?>
<squares>
  <!-- f(x) = x^2 -->
  <data x="1" y="1"/>
  <data x="2" y="4"/>
  <data x="3" y="9"/>
  <data x="4" y="16"/>
  <data x="5" y="25"/>
</squares>
```

See the [Usage](https://github.com/oozcitak/xmlbuilder-js/wiki/Usage) page in the wiki for more detailed instructions.

### License:

`xmlbuilder-js` is [MIT Licensed](http://opensource.org/licenses/mit-license.php).
