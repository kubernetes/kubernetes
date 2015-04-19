[![Build Status][1]][2] [![dependency status][9]][10] [![dev dependency status][11]][12]

# extend() for Node.js <sup>[![Version Badge][8]][3]</sup>

`node-extend` is a port of the classic extend() method from jQuery. It behaves as you expect. It is simple, tried and true.

## Installation

This package is available on [npm][3] as: `extend`

``` sh
npm install extend
```

## Usage

**Syntax:** extend **(** [`deep`], `target`, `object1`, [`objectN`] **)** 

*Extend one object with one or more others, returning the modified object.*

Keep in mind that the target object will be modified, and will be returned from extend().

If a boolean true is specified as the first argument, extend performs a deep copy, recursively copying any objects it finds. Otherwise, the copy will share structure with the original object(s).
Undefined properties are not copied. However, properties inherited from the object's prototype will be copied over.

### Arguments

* `deep` *Boolean* (optional)  
If set, the merge becomes recursive (i.e. deep copy).
* `target`	*Object*  
The object to extend.
* `object1`	*Object*  
The object that will be merged into the first.
* `objectN` *Object* (Optional)  
More objects to merge into the first.

## License

`node-extend` is licensed under the [MIT License][4].

## Acknowledgements

All credit to the jQuery authors for perfecting this amazing utility.

Ported to Node.js by [Stefan Thomas][5] with contributions by [Jonathan Buchanan][6] and [Jordan Harband][7].

[1]: https://travis-ci.org/justmoon/node-extend.png
[2]: https://travis-ci.org/justmoon/node-extend
[3]: https://npmjs.org/package/extend
[4]: http://opensource.org/licenses/MIT
[5]: https://github.com/justmoon
[6]: https://github.com/insin
[7]: https://github.com/ljharb
[8]: http://vb.teelaun.ch/justmoon/node-extend.svg
[9]: https://david-dm.org/justmoon/node-extend.png
[10]: https://david-dm.org/justmoon/node-extend
[11]: https://david-dm.org/justmoon/node-extend/dev-status.png
[12]: https://david-dm.org/justmoon/node-extend#info=devDependencies

