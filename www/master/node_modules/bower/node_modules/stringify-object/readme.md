# stringify-object [![Build Status](https://secure.travis-ci.org/yeoman/stringify-object.svg?branch=master)](http://travis-ci.org/yeoman/stringify-object)

> Stringify an object/array like JSON.stringify just without all the double-quotes.

Useful for when you want to get the string representation of an object in a formatted way.

It also handles circular references and lets you specify quote type.


## Install

```sh
$ npm install --save stringify-object
```


## Usage

```js
var obj = {
    foo: 'bar',
    'arr': [1, 2, 3],
    nested: { hello: "world" }
};

var pretty = stringifyObject(obj, {
    indent: '  ',
    singleQuotes: false
});

console.log(pretty);
/*
{
  foo: "bar",
  arr: [
    1,
    2,
    3
  ],
  nested: {
    hello: "world"
  }
}
*/
```


## API

### stringifyObject(object, [options])

Accepts an object to stringify and optionally an option object. Circular references will be replaced with `null`.

#### options

##### indent

Type: `string`  
Default: `'\t'`

Choose the indentation you prefer.

##### singleQuotes

Type: `boolean`  
Default: `true`

Set to false to get double-quoted strings.


## License

[BSD license](http://opensource.org/licenses/bsd-license.php) and copyright Google
