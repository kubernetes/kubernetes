// Copyright 2014 Selenium committers
// Copyright 2014 Software Freedom Conservancy
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

/** @fileoverview Utilities for working with Firefox extensions. */

'use strict';

var AdmZip = require('adm-zip'),
    fs = require('fs'),
    path = require('path'),
    util = require('util'),
    xml = require('xml2js');

var promise = require('..').promise,
    checkedCall = promise.checkedNodeCall,
    io = require('../io');


/**
 * Thrown when there an add-on is malformed.
 * @param {string} msg The error message.
 * @constructor
 * @extends {Error}
 */
function AddonFormatError(msg) {
  Error.call(this);

  Error.captureStackTrace(this, AddonFormatError);

  /** @override */
  this.name = AddonFormatError.name;

  /** @override */
  this.message = msg;
}
util.inherits(AddonFormatError, Error);



/**
 * Installs an extension to the given directory.
 * @param {string} extension Path to the extension to install, as either a xpi
 *     file or a directory.
 * @param {string} dir Path to the directory to install the extension in.
 * @return {!promise.Promise.<string>} A promise for the add-on ID once
 *     installed.
 */
function install(extension, dir) {
  return getDetails(extension).then(function(details) {
    function returnId() { return details.id; }

    var dst = path.join(dir, details.id);
    if (extension.slice(-4) === '.xpi') {
      if (!details.unpack) {
        return io.copy(extension, dst + '.xpi').then(returnId);
      } else {
        return checkedCall(fs.readFile, extension).then(function(buff) {
          var zip = new AdmZip(buff);
          // TODO: find an async library for inflating a zip archive.
          new AdmZip(buff).extractAllTo(dst, true);
        }).then(returnId);
      }
    } else {
      return io.copyDir(extension, dst).then(returnId);
    }
  });
}


/**
 * Describes a Firefox add-on.
 * @typedef {{id: string, name: string, version: string, unpack: boolean}}
 */
var AddonDetails;


/**
 * Extracts the details needed to install an add-on.
 * @param {string} addonPath Path to the extension directory.
 * @return {!promise.Promise.<!AddonDetails>} A promise for the add-on details.
 */
function getDetails(addonPath) {
  return readManifest(addonPath).then(function(doc) {
    var em = getNamespaceId(doc, 'http://www.mozilla.org/2004/em-rdf#');
    var rdf = getNamespaceId(
        doc, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#');

    var description = doc[rdf + 'RDF'][rdf + 'Description'][0];
    var details = {
      id: getNodeText(description, em + 'id'),
      name: getNodeText(description, em + 'name'),
      version: getNodeText(description, em + 'version'),
      unpack: getNodeText(description, em + 'unpack') || false
    };

    if (typeof details.unpack === 'string') {
      details.unpack = details.unpack.toLowerCase() === 'true';
    }

    if (!details.id) {
      throw new AddonFormatError('Could not find add-on ID for ' + addonPath);
    }

    return details;
  });

  function getNodeText(node, name) {
    return node[name] && node[name][0] || '';
  }

  function getNamespaceId(doc, url) {
    var keys = Object.keys(doc);
    if (keys.length !== 1) {
      throw new AddonFormatError('Malformed manifest for add-on ' + addonPath);
    }

    var namespaces = doc[keys[0]].$;
    var id = '';
    Object.keys(namespaces).some(function(ns) {
      if (namespaces[ns] !== url) {
        return false;
      }

      if (ns.indexOf(':') != -1) {
        id = ns.split(':')[1] + ':';
      }
      return true;
    });
    return id;
  }
}


/**
 * Reads the manifest for a Firefox add-on.
 * @param {string} addonPath Path to a Firefox add-on as a xpi or an extension.
 * @return {!promise.Promise.<!Object>} A promise for the parsed manifest.
 */
function readManifest(addonPath) {
  var manifest;

  if (addonPath.slice(-4) === '.xpi') {
    manifest = checkedCall(fs.readFile, addonPath).then(function(buff) {
      var zip = new AdmZip(buff);
      if (!zip.getEntry('install.rdf')) {
        throw new AddonFormatError(
            'Could not find install.rdf in ' + addonPath);
      }
      var done = promise.defer();
      zip.readAsTextAsync('install.rdf', done.fulfill);
      return done.promise;
    });
  } else {
    manifest = checkedCall(fs.stat, addonPath).then(function(stats) {
      if (!stats.isDirectory()) {
        throw Error(
            'Add-on path is niether a xpi nor a directory: ' + addonPath);
      }
      return checkedCall(fs.readFile, path.join(addonPath, 'install.rdf'));
    });
  }

  return manifest.then(function(content) {
    return checkedCall(xml.parseString, content);
  });
}


// PUBLIC API


exports.install = install;
