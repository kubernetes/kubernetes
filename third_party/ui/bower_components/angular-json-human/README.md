angular-json-human [![Analytics](https://ga-beacon.appspot.com/UA-2694988-7/angular-json-human/readme?pixel)](https://github.com/yaru22/angular-json-human)
==================
Angular directive to convert JSON into human readable table. Inspired by https://github.com/marianoguerra/json.human.js.

Demo
----
Check out the demo [here](http://www.brianpark.ca/projects/angular_json_human/demo/).

Dependency
----------
This directive requires `lodash`. I'm going to remove the dependency in the future release.

How to Use
----------
Install it via `bower`:
```
$ bower install angular-json-human
```

Include `angular-json-human.(js|css)` in your project. Load the directive after loading `angular.js`

```
<script src="<path to angular.js>"></script>
<script src="<path to angular-json-human.js>"></script>
```

Specify angular-json-human as a dependency of your Angular module.

```
var app = angular.module('ngApp', [
  'yaru22.jsonHuman'
]);
```

Use it in your project.

```
<html ng-app="ngApp">
...
<body ng-controller="MainCtrl">
  <div json-human="jsonStr"></div>
  ...
</body>
</html>
```

or check out my [Plunker](http://plnkr.co/edit/0wEPmUsw5kKbBo9RjXW4?p=preview) for the minimal setup.


How to Contribute
-----------------
```
$ git clone https://github.com/yaru22/angular-json-human.git
$ cd angular-json-human
$ npm install; bower install
$ # modify the source code in src/
$ grunt clean; grunt build
$ # test your changes; you can modify demo/ and serve it locally to see the changes.
$ # submit a pull request
```

TODO
----
- Remove the dependency on Lodash.
