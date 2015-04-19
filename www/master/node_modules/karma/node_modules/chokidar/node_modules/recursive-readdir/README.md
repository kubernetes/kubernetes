#recursive-readdir

A simple Node module for recursively listing all files in a directory,
or in any subdirectories.

It does not list directories themselves.

##Installation

    npm install recursive-readdir

##Usage


```javascript
var recursive-readdir = require('recursive-readdir');

recursive-readdir('some/path', function (err, files) {
  // Files is an array of filename
  console.log(files);
});
```
