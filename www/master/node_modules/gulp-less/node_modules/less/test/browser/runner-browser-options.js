var less = {};

// There originally run inside describe method. However, since they have not
// been inside it, they run at jasmine compile time (not runtime). It all
// worked cause less.js was in async mode and custom phantom runner had
// different setup then grunt-contrib-jasmine. They have been created before
// less.js run, even as they have been defined in spec.

// test inline less in style tags by grabbing an assortment of less files and doing `@import`s
var testFiles = ['charsets', 'colors', 'comments', 'css-3', 'strings', 'media', 'mixins'],
    testSheets = [];

// setup style tags with less and link tags pointing to expected css output
for (var i = 0; i < testFiles.length; i++) {
  var file = testFiles[i],
      lessPath  = '/test/less/' + file + '.less',
      cssPath   = '/test/css/' + file + '.css',
      lessStyle = document.createElement('style'),
      cssLink   = document.createElement('link'),
      lessText  = '@import "' + lessPath + '";';

  lessStyle.type = 'text/less';
  lessStyle.id = file;
  lessStyle.href = file;

  if (lessStyle.styleSheet) {
    lessStyle.styleSheet.cssText = lessText;
  } else {
    lessStyle.innerHTML = lessText;
  }

  cssLink.rel = 'stylesheet';
  cssLink.type = 'text/css';
  cssLink.href = cssPath;
  cssLink.id = 'expected-' + file;

  var head = document.getElementsByTagName('head')[0];

  head.appendChild(lessStyle);
  head.appendChild(cssLink);
  testSheets[i] = lessStyle;
}