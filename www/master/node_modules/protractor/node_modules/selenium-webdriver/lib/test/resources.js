// Copyright 2013 Selenium committers
// Copyright 2013 Software Freedom Conservancy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

'use strict';

var fs = require('fs'),
    path = require('path');

var resourceRoot = require('../../_base').isDevMode() ?
    require('./build').projectRoot() :
    path.join(__dirname, 'data');


// PUBLIC API


/**
 * Locates a test resource.
 * @param {string} resourcePath Path of the resource to locate.
 * @param {string} filePath The file to locate from the root of the project.
 * @return {string} The full path for the file, if it exists.
 * @throws {Error} If the file does not exist.
 */
exports.locate = function(filePath) {
  var fullPath = path.normalize(path.join(resourceRoot, filePath));
  if (!fs.existsSync(fullPath)) {
    throw Error('File does not exist: ' + filePath);
  }
  return fullPath;
};
