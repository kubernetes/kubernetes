# qs

A querystring parsing and stringifying library with some added security.

[![Build Status](https://secure.travis-ci.org/hapijs/qs.svg)](http://travis-ci.org/hapijs/qs)

Lead Maintainer: [Nathan LaFreniere](https://github.com/nlf)

The **qs** module was original created and maintained by [TJ Holowaychuk](https://github.com/visionmedia/node-querystring).

## Usage

```javascript
var Qs = require('qs');

var obj = Qs.parse('a=c');    // { a: 'c' }
var str = Qs.stringify(obj);  // 'a=c'
```

### Objects

**qs** allows you to create nested objects within your query strings, by surrounding the name of sub-keys with square brackets `[]`.
For example, the string `'foo[bar]=baz'` converts to:

```javascript
{
  foo: {
    bar: 'baz'
  }
}
```

You can also nest your objects, like `'foo[bar][baz]=foobarbaz'`:

```javascript
{
  foo: {
    bar: {
      baz: 'foobarbaz'
    }
  }
}
```

By default, when nesting objects **qs** will only parse up to 5 children deep. This means if you attempt to parse a string like
`'a[b][c][d][e][f][g][h][i]=j'` your resulting object will be:

```javascript
{
  a: {
    b: {
      c: {
        d: {
          e: {
            f: {
              '[g][h][i]': 'j'
            }
          }
        }
      }
    }
  }
}
```

This depth can be overridden by passing a `depth` option to `Qs.parse(string, depth)`:

```javascript
Qs.parse('a[b][c][d][e][f][g][h][i]=j', 1);
// { a: { b: { '[c][d][e][f][g][h][i]': 'j' } } }
```

The depth limit mitigate abuse when **qs** is used to parse user input, and it is recommended to keep it a reasonably small number.

### Arrays

**qs** can also parse arrays using a similar `[]` notation:

```javascript
Qs.parse('a[]=b&a[]=c');
// { a: ['b', 'c'] }
```

You may specify an index as well:

```javascript
Qs.parse('a[1]=c&a[0]=b');
// { a: ['b', 'c'] }
```

Note that the only difference between an index in an array and a key in an object is that the value between the brackets must be a number
to create an array. When creating arrays with specific indices, **qs** will compact a sparse array to only the existing values preserving
their order:

```javascript
Qs.parse('a[1]=b&a[15]=c');
// { a: ['b', 'c'] }
```

**qs** will also limit specifying indices in an array to a maximum index of `20`. Any array members with an index of greater than `20` will
instead be converted to an object with the index as the key:

```javascript
Qs.parse('a[100]=b');
// { a: { '100': 'b' } }
```

If you mix notations, **qs** will merge the two items into an object:

```javascript
Qs.parse('a[0]=b&a[b]=c');
// { a: { '0': 'b', b: 'c' } }
```

You can also create arrays of objects:

```javascript
Qs.parse('a[][b]=c');
// { a: [{ b: 'c' }] }
```