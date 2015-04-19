# package-json [![Build Status](https://travis-ci.org/sindresorhus/package-json.svg?branch=master)](https://travis-ci.org/sindresorhus/package-json)

> Get the package.json of a package from the npm registry


## Install

```
$ npm install --save package-json
```


## Usage

```js
var packageJson = require('package-json');

packageJson('pageres', 'latest', function (err, json) {
	console.log(json);
	//=> { name: 'pageres', ... }
});

// also works with scoped packages
packageJson('@company/package', 'latest', function (err, json) {
	console.log(json);
	//=> { name: 'package', ... }
});
```


## API

### packageJson(name, [version], callback)

You can optionally specify a version (e.g. `0.1.0`) or `latest`.  
If you don't specify a version you'll get the [main entry](http://registry.npmjs.org/pageres/) containing all versions.


## Related

- [`npm-keyword`](https://github.com/sindresorhus/npm-keyword) - Get a list of npm packages with a certain keyword


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
