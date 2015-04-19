node-html-entities
==================

Fast html entities library.


Installation
------------

    npm install html-entities

Usage
-----

####XML entities####

HTML validity and XSS attack prevention you can achieve from XmlEntities class.

```javascript
var Entities = require('html-entities').XmlEntities;

entities = new Entities();

console.log(entities.encode('<>"\'&©®')); // &lt;&gt;&quot;&apos;&amp;©®
console.log(entities.encodeNonUTF('<>"\'&©®')); // &lt;&gt;&quot;&apos;&amp;&#169;&#174;
console.log(entities.encodeNonASCII('<>"\'&©®')); // <>"\'&©®
console.log(entities.decode('&lt;&gt;&quot;&apos;&amp;&copy;&reg;&#8710;')); // <>"'&&copy;&reg;∆
```

####All HTML entities encoding/decoding####


```javascript
var Entities = require('html-entities').AllHtmlEntities;

entities = new Entities();

console.log(entities.encode('<>"&©®∆')); // &lt;&gt;&quot;&amp;&copy;&reg;∆
console.log(entities.encodeNonUTF('<>"&©®∆')); // &lt;&gt;&quot;&amp;&copy;&reg;&#8710;
console.log(entities.encodeNonASCII('<>"&©®∆')); // <>"&©®&#8710;
console.log(entities.decode('&lt;&gt;&quot;&amp;&copy;&reg;')); // <>"&©®
```

####Available classes####

```javascript
var XmlEntities = require('html-entities').XmlEntities, // <>"'& + &#...; decoding
    Html4Entities = require('html-entities').Html4Entities, // HTML4 entities.
    Html5Entities = require('html-entities').Html5Entities, // HTML5 entities.
    AllHtmlEntities = require('html-entities').AllHtmlEntities; // Synonym for HTML5 entities.
```

Supports four methods for every class:

* encode — encodes, replacing characters to its entity representations. Ignores UTF characters with no entity representation.
* encodeNonUTF — encodes, replacing characters to its entity representations. Inserts numeric entities for UTF characters.
* encodeNonASCII — encodes, replacing only non-ASCII characters to its numeric entity representations.
* decode — decodes, replacing entities to characters. Unknown entities are left as is.
