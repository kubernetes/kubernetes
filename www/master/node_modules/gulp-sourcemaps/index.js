'use strict';
var through = require('through2');
var fs = require('graceful-fs');
var path = require('path');
var File = require('vinyl');
var convert = require('convert-source-map');
var stripBom = require('strip-bom');

var PLUGIN_NAME = 'gulp-sourcemap';
var urlRegex = /^https?:\/\//;

/**
 * Initialize source mapping chain
 */
module.exports.init = function init(options) {
  function sourceMapInit(file, encoding, callback) {
    /*jshint validthis:true */

    if (file.isNull()) {
      this.push(file);
      return callback();
    }

    if (file.isStream()) {
      return callback(new Error(PLUGIN_NAME + '-init: Streaming not supported'));
    }

    var fileContent = file.contents.toString();
    var sourceMap;

    if (options && options.loadMaps) {
      var sourcePath = ''; //root path for the sources in the map

      // Try to read inline source map
      sourceMap = convert.fromSource(fileContent);
      if (sourceMap) {
        sourceMap = sourceMap.toObject();
        // sources in map are relative to the source file
        sourcePath = path.dirname(file.path);
        fileContent = convert.removeComments(fileContent);
      } else {
        // look for source map comment referencing a source map file
        var mapComment = convert.mapFileCommentRegex.exec(fileContent);

        var mapFile;
        if (mapComment) {
          mapFile = path.resolve(path.dirname(file.path), mapComment[1] || mapComment[2]);
          fileContent = convert.removeMapFileComments(fileContent);
        // if no comment try map file with same name as source file
        } else {
          mapFile = file.path + '.map';
        }

        // sources in external map are relative to map file
        sourcePath = path.dirname(mapFile);

        try {
          sourceMap = JSON.parse(stripBom(fs.readFileSync(mapFile, 'utf8')));
        } catch(e) {}
      }

      // fix source paths and sourceContent for imported source map
      if (sourceMap) {
        sourceMap.sourcesContent = sourceMap.sourcesContent || [];
        sourceMap.sources.forEach(function(source, i) {
          if (source.match(urlRegex)) {
            sourceMap.sourcesContent[i] = sourceMap.sourcesContent[i] || null;
            return;
          }
          var absPath = path.resolve(sourcePath, source);
          sourceMap.sources[i] = unixStylePath(path.relative(file.base, absPath));

          if (!sourceMap.sourcesContent[i]) {
            var sourceContent = null;
            if (sourceMap.sourceRoot) {
              if (sourceMap.sourceRoot.match(urlRegex)) {
                sourceMap.sourcesContent[i] = null;
                return;
              }
              absPath = path.resolve(sourcePath, sourceMap.sourceRoot, source);
            }

            // if current file: use content
            if (absPath === file.path) {
              sourceContent = fileContent;

            // else load content from file
            } else {
              try {
                if (options.debug)
                  console.log(PLUGIN_NAME + '-init: No source content for "' + source + '". Loading from file.');
                sourceContent = stripBom(fs.readFileSync(absPath, 'utf8'));
              } catch (e) {
                if (options.debug)
                  console.warn(PLUGIN_NAME + '-init: source file not found: ' + absPath);
              }
            }
            sourceMap.sourcesContent[i] = sourceContent;
          }
        });

        // remove source map comment from source
        file.contents = new Buffer(fileContent, 'utf8');
      }
    }

    if (!sourceMap) {
      // Make an empty source map
      sourceMap = {
        version : 3,
        names: [],
        mappings: '',
        sources: [unixStylePath(file.relative)],
        sourcesContent: [fileContent]
      };
    }

    sourceMap.file = unixStylePath(file.relative);
    file.sourceMap = sourceMap;

    this.push(file);
    callback();
  }

  return through.obj(sourceMapInit);
};

/**
 * Write the source map
 *
 * @param options options to change the way the source map is written
 *
 */
module.exports.write = function write(destPath, options) {
  if (options === undefined && Object.prototype.toString.call(destPath) === '[object Object]') {
    options = destPath;
    destPath = undefined;
  }
  options = options || {};

  // set defaults for options if unset
  if (options.includeContent === undefined)
    options.includeContent = true;
  if (options.addComment === undefined)
    options.addComment = true;

  function sourceMapWrite(file, encoding, callback) {
    /*jshint validthis:true */

    if (file.isNull() || !file.sourceMap) {
      this.push(file);
      return callback();
    }

    if (file.isStream()) {
      return callback(new Error(PLUGIN_NAME + '-write: Streaming not supported'));
    }

    var sourceMap = file.sourceMap;
    // fix paths if Windows style paths
    sourceMap.file = unixStylePath(file.relative);
    sourceMap.sources = sourceMap.sources.map(function(filePath) {
      return unixStylePath(filePath);
    });

    if (options.sourceRoot) {
      if (typeof options.sourceRoot === 'function') {
        sourceMap.sourceRoot = options.sourceRoot(file);
      } else {
        sourceMap.sourceRoot = options.sourceRoot;
      }
    }

    if (options.includeContent) {
      sourceMap.sourcesContent = sourceMap.sourcesContent || [];

      // load missing source content
      for (var i = 0; i < file.sourceMap.sources.length; i++) {
        if (!sourceMap.sourcesContent[i]) {
          var sourcePath = path.resolve(sourceMap.sourceRoot || file.base, sourceMap.sources[i]);
          try {
            if (options.debug)
              console.log(PLUGIN_NAME + '-write: No source content for "' + sourceMap.sources[i] + '". Loading from file.');
            sourceMap.sourcesContent[i] = stripBom(fs.readFileSync(sourcePath, 'utf8'));
          } catch (e) {
            if (options.debug)
              console.warn(PLUGIN_NAME + '-write: source file not found: ' + sourcePath);
          }
        }
      }
      sourceMap.sourceRoot = sourceMap.sourceRoot || '/source/';
    } else {
      delete sourceMap.sourcesContent;
    }

    var extension = file.relative.split('.').pop();
    var commentFormatter;

    switch (extension) {
      case 'css':
        commentFormatter = function(url) { return "\n/*# sourceMappingURL=" + url + " */"; };
        break;
      case 'js':
        commentFormatter = function(url) { return "\n//# sourceMappingURL=" + url; };
        break;
      default:
        commentFormatter = function(url) { return ""; };
    }

    var comment, sourceMappingURLPrefix;
    if (!destPath) {
      // encode source map into comment
      var base64Map = new Buffer(JSON.stringify(sourceMap)).toString('base64');
      comment = commentFormatter('data:application/json;base64,' + base64Map);
    } else {
      // add new source map file to stream
      var sourceMapFile = new File({
        cwd: file.cwd,
        base: file.base,
        path: path.join(file.base, destPath, file.relative) + '.map',
        contents: new Buffer(JSON.stringify(sourceMap))
      });
      this.push(sourceMapFile);

      comment = commentFormatter(unixStylePath(path.join(path.relative(path.dirname(file.path), file.base), destPath, file.relative) + '.map'));

      if (options.sourceMappingURLPrefix) {
        if (typeof options.sourceMappingURLPrefix === 'function') {
          sourceMappingURLPrefix = options.sourceMappingURLPrefix(file);
        } else {
          sourceMappingURLPrefix = options.sourceMappingURLPrefix;
        }
        comment = comment.replace(/sourceMappingURL=\.*/, 'sourceMappingURL=' + sourceMappingURLPrefix);
      }
    }

    // append source map comment
    if (options.addComment)
      file.contents = Buffer.concat([file.contents, new Buffer(comment)]);

    this.push(file);
    callback();
  }

  return through.obj(sourceMapWrite);
};

function unixStylePath(filePath) {
  return filePath.split(path.sep).join('/');
}
