# ent

Encode and decode HTML entities

[![browser support](http://ci.testling.com/substack/node-ent.png)](http://ci.testling.com/substack/node-ent)

[![build status](https://secure.travis-ci.org/substack/node-ent.png)](http://travis-ci.org/substack/node-ent)

# example

``` js
var ent = require('ent');
console.log(ent.encode('<span>©moo</span>'))
console.log(ent.decode('&pi; &amp; &rho;'));
```

```
&lt;span&gt;&copy;moo&lt;/span&gt;
π & ρ
```

![ent](http://substack.net/images/ent.png)

# methods

## encode(str)

Escape unsafe characters in `str` with html entities.

## decode(str)

Convert html entities in `str` back to raw text.

# credits

HTML entity tables shamelessly lifted from perl's
[HTML::Entities](http://cpansearch.perl.org/src/GAAS/HTML-Parser-3.68/lib/HTML/Entities.pm)

# install

With [npm](https://npmjs.org) do:

```
npm install ent
```

# license

MIT
