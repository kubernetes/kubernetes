module.exports = require("./jasmine-core/jasmine.js");
module.exports.boot = require('./jasmine-core/node_boot.js');

var path = require('path'),
    fs = require('fs');

var rootPath = path.join(__dirname, "jasmine-core"),
    bootFiles = ['boot.js'],
    nodeBootFiles = ['node_boot.js'],
    cssFiles = [],
    jsFiles = [],
    jsFilesToSkip = ['jasmine.js'].concat(bootFiles, nodeBootFiles);

fs.readdirSync(rootPath).forEach(function(file) {
  if(fs.statSync(path.join(rootPath, file)).isFile()) {
    switch(path.extname(file)) {
      case '.css':
        cssFiles.push(file);
      break;
      case '.js':
        if (jsFilesToSkip.indexOf(file) < 0) {
        jsFiles.push(file);
      }
      break;
    }
  }
});

module.exports.files = {
  path: rootPath,
  bootDir: rootPath,
  bootFiles: bootFiles,
  nodeBootFiles: nodeBootFiles,
  cssFiles: cssFiles,
  jsFiles: ['jasmine.js'].concat(jsFiles),
  imagesDir: path.join(__dirname, '../images')
};
