<a href="https://github.com/spumko"><img src="https://raw.github.com/spumko/spumko/master/images/from.png" align="right" /></a>
![hoek Logo](https://raw.github.com/spumko/hoek/master/images/hoek.png)

General purpose node utilities

[![Build Status](https://secure.travis-ci.org/spumko/hoek.png)](http://travis-ci.org/spumko/hoek)

# Table of Contents

* [Introduction](#introduction "Introduction")
* [Object](#object "Object")
  * [clone](#cloneobj "clone")
  * [merge](#mergetarget-source-isnulloverride-ismergearrays "merge")
  * [applyToDefaults](#applytodefaultsdefaults-options "applyToDefaults")
  * [unique](#uniquearray-key "unique")
  * [mapToObject](#maptoobjectarray-key "mapToObject")
  * [intersect](#intersectarray1-array2 "intersect")
  * [matchKeys](#matchkeysobj-keys "matchKeys")
  * [flatten](#flattenarray-target "flatten")
  * [removeKeys](#removekeysobject-keys "removeKeys")
  * [reach](#reachobj-chain "reach")
  * [inheritAsync](#inheritasyncself-obj-keys "inheritAsync")
  * [rename](#renameobj-from-to "rename")
* [Timer](#timer "Timer")
* [Binary Encoding/Decoding](#binary "Binary Encoding/Decoding")
  * [base64urlEncode](#binary64urlEncodevalue "binary64urlEncode")
  * [base64urlDecode](#binary64urlDecodevalue "binary64urlDecode")
* [Escaping Characters](#escaped "Escaping Characters")
  * [escapeHtml](#escapeHtmlstring "escapeHtml")
  * [escapeHeaderAttribute](#escapeHeaderAttributeattribute "escapeHeaderAttribute")
  * [escapeRegex](#escapeRegexstring "escapeRegex")
* [Errors](#errors "Errors")
  * [assert](#assertmessage "assert")
  * [abort](#abortmessage "abort")
  * [displayStack](#displayStackslice "displayStack")
  * [callStack](#callStackslice "callStack")
  * [toss](#tosscondition "toss")
* [Load files](#load-files "Load Files")
  * [loadPackage](#loadPackagedir "loadpackage")
  * [loadDirModules](#loadDirModulespath-excludefiles-target "loaddirmodules")



# Introduction

The *Hoek* general purpose node utilities library is used to aid in a variety of manners. It comes with useful methods for Arrays (clone, merge, applyToDefaults), Objects (removeKeys, copy), Asserting and more. 

For example, to use Hoek to set configuration with default options:
```javascript
var Hoek = require('hoek');

var default = {url : "www.github.com", port : "8000", debug : true}

var config = Hoek.applyToDefaults(default, {port : "3000", admin : true});

// In this case, config would be { url: 'www.github.com', port: '3000', debug: true, admin: true }
```

Under each of the sections (such as Array), there are subsections which correspond to Hoek methods. Each subsection will explain how to use the corresponding method. In each js excerpt below, the var Hoek = require('hoek') is omitted for brevity.

## Object

Hoek provides several helpful methods for objects and arrays.

### clone(obj)

This method is used to clone an object or an array. A *deep copy* is made (duplicates everything, including values that are objects). 

```javascript

var nestedObj = {
        w: /^something$/ig,
        x: {
            a: [1, 2, 3],
            b: 123456,
            c: new Date()
        },
        y: 'y',
        z: new Date()
    };

var copy = Hoek.clone(nestedObj);

copy.x.b = 100;

console.log(copy.y)        // results in 'y'
console.log(nestedObj.x.b) // results in 123456
console.log(copy.x.b)      // results in 100
```

### merge(target, source, isNullOverride, isMergeArrays)
isNullOverride, isMergeArrays default to true

Merge all the properties of source into target, source wins in conflic, and by default null and undefined from source are applied


```javascript

var target = {a: 1, b : 2}
var source = {a: 0, c: 5}
var source2 = {a: null, c: 5}

var targetArray = [1, 2, 3];
var sourceArray = [4, 5];

var newTarget = Hoek.merge(target, source);     // results in {a: 0, b: 2, c: 5}
newTarget = Hoek.merge(target, source2);        // results in {a: null, b: 2, c: 5}
newTarget = Hoek.merge(target, source2, false); // results in {a: 1, b: 2, c: 5}

newTarget = Hoek.merge(targetArray, sourceArray)              // results in [1, 2, 3, 4, 5]
newTarget = Hoek.merge(targetArray, sourceArray, true, false) // results in [4, 5]




```

### applyToDefaults(defaults, options)

Apply options to a copy of the defaults

```javascript

var defaults = {host: "localhost", port: 8000};
var options = {port: 8080};

var config = Hoek.applyToDefaults(defaults, options); // results in {host: "localhost", port: 8080};


```

### unique(array, key)

Remove duplicate items from Array

```javascript

var array = [1, 2, 2, 3, 3, 4, 5, 6];

var newArray = Hoek.unique(array); // results in [1,2,3,4,5,6];

array = [{id: 1}, {id: 1}, {id: 2}];

newArray = Hoek.unique(array, "id") // results in [{id: 1}, {id: 2}]

```

### mapToObject(array, key)

Convert an Array into an Object

```javascript

var array = [1,2,3];
var newObject = Hoek.mapToObject(array); // results in [{"1": true}, {"2": true}, {"3": true}]

array = [{id: 1}, {id: 2}];
newObject = Hoek.mapToObject(array, "id") // results in [{"id": 1}, {"id": 2}]

```
### intersect(array1, array2)

Find the common unique items in two arrays

```javascript

var array1 = [1, 2, 3];
var array2 = [1, 4, 5];

var newArray = Hoek.intersect(array1, array2) // results in [1]

```

### matchKeys(obj, keys) 

Find which keys are present

```javascript

var obj = {a: 1, b: 2, c: 3};
var keys = ["a", "e"];

Hoek.matchKeys(obj, keys) // returns ["a"]

```

### flatten(array, target)

Flatten an array

```javascript

var array = [1, 2, 3];
var target = [4, 5]; 

var flattenedArray = Hoek.flatten(array, target) // results in [4, 5, 1, 2, 3];

```

### removeKeys(object, keys)

Remove keys

```javascript

var object = {a: 1, b: 2, c: 3, d: 4};

var keys = ["a", "b"];

Hoek.removeKeys(object, keys) // object is now {c: 3, d: 4}

```

### reach(obj, chain)

Converts an object key chain string to reference

```javascript

var chain = 'a.b.c';
var obj = {a : {b : { c : 1}}};

Hoek.reach(obj, chain) // returns 1

```

### inheritAsync(self, obj, keys) 

Inherits a selected set of methods from an object, wrapping functions in asynchronous syntax and catching errors

```javascript

var targetFunc = function () { };

var proto = {
                a: function () {
                    return 'a!';
                },
                b: function () {
                    return 'b!';
                },
                c: function () {
                    throw new Error('c!');
                }
            };

var keys = ['a', 'c'];

Hoek.inheritAsync(targetFunc, proto, ['a', 'c']);

var target = new targetFunc();

target.a(function(err, result){console.log(result)}         // returns 'a!'       

target.c(function(err, result){console.log(result)}         // returns undefined

target.b(function(err, result){console.log(result)}         // gives error: Object [object Object] has no method 'b'

```

### rename(obj, from, to)

Rename a key of an object

```javascript

var obj = {a : 1, b : 2};

Hoek.rename(obj, "a", "c");     // obj is now {c : 1, b : 2}

```


# Timer

A Timer object. Initializing a new timer object sets the ts to the number of milliseconds elapsed since 1 January 1970 00:00:00 UTC.

```javascript


example : 


var timerObj = new Hoek.Timer();
console.log("Time is now: " + timerObj.ts)
console.log("Elapsed time from initialization: " + timerObj.elapsed() + 'milliseconds')

```

# Binary Encoding/Decoding

### base64urlEncode(value)

Encodes value in Base64 or URL encoding

### base64urlDecode(value)

Decodes data in Base64 or URL encoding.
# Escaping Characters

Hoek provides convenient methods for escaping html characters. The escaped characters are as followed:

```javascript

internals.htmlEscaped = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#x27;',
    '`': '&#x60;'
};

```

### escapeHtml(string)

```javascript

var string = '<html> hey </html>';
var escapedString = Hoek.escapeHtml(string); // returns &lt;html&gt; hey &lt;/html&gt;

```

### escapeHeaderAttribute(attribute)

Escape attribute value for use in HTTP header

```javascript

var a = Hoek.escapeHeaderAttribute('I said "go w\\o me"');  //returns I said \"go w\\o me\"


```


### escapeRegex(string)

Escape string for Regex construction

```javascript

var a = Hoek.escapeRegex('4^f$s.4*5+-_?%=#!:@|~\\/`"(>)[<]d{}s,');  // returns 4\^f\$s\.4\*5\+\-_\?%\=#\!\:@\|~\\\/`"\(>\)\[<\]d\{\}s\,



```

# Errors

### assert(message)

```javascript

var a = 1, b =2;

Hoek.assert(a === b, 'a should equal b');  // ABORT: a should equal b

```

### abort(message)

First checks if process.env.NODE_ENV === 'test', and if so, throws error message. Otherwise,
displays most recent stack and then exits process.



### displayStack(slice)

Displays the trace stack

```javascript

var stack = Hoek.displayStack();
console.log(stack) // returns something like:

[ 'null (/Users/user/Desktop/hoek/test.js:4:18)',
  'Module._compile (module.js:449:26)',
  'Module._extensions..js (module.js:467:10)',
  'Module.load (module.js:356:32)',
  'Module._load (module.js:312:12)',
  'Module.runMain (module.js:492:10)',
  'startup.processNextTick.process._tickCallback (node.js:244:9)' ]

```

### callStack(slice)

Returns a trace stack array.

```javascript

var stack = Hoek.callStack();
console.log(stack)  // returns something like:

[ [ '/Users/user/Desktop/hoek/test.js', 4, 18, null, false ],
  [ 'module.js', 449, 26, 'Module._compile', false ],
  [ 'module.js', 467, 10, 'Module._extensions..js', false ],
  [ 'module.js', 356, 32, 'Module.load', false ],
  [ 'module.js', 312, 12, 'Module._load', false ],
  [ 'module.js', 492, 10, 'Module.runMain', false ],
  [ 'node.js',
    244,
    9,
    'startup.processNextTick.process._tickCallback',
    false ] ]


```

### toss(condition)

toss(condition /*, [message], callback */)

Return an error as first argument of a callback


# Load Files

### loadPackage(dir)

Load and parse package.json process root or given directory

```javascript

var pack = Hoek.loadPackage();  // pack.name === 'hoek'

```

### loadDirModules(path, excludeFiles, target) 

Loads modules from a given path; option to exclude files (array).




