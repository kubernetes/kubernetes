'use strict';

/**
 * Build in Native modules.
 */
var path = require('path')
  , fs = require('fs')
  , vm = require('vm');

/**
 * Third party modules.
 */
var request = require('request')
  , yaml = require('yamlparser');

exports.update = function update(callback) {
  // Fetch the remote resource as that is frequently updated
  request(exports.remote, function downloading(err, res, remote) {
    if (err) return callback(err);
    if (res.statusCode !== 200) return callback(new Error('Invalid statusCode returned'));

    // Also get some local additions that are missing from the source
    fs.readFile(exports.local, 'utf8', function reading(err, local) {
      if (err) return callback(err);

      // Parse the contents
      exports.parse([ remote, local ], function parsing(err, results, source) {
        callback(err, results);

        if (source && !err) {
          fs.writeFile(exports.output, source, function idk(err) {
            if (err) {
              console.error('Failed to save the generated file due to reasons', err);
            }
          });
        }
      });
    });
  });
};

exports.parse = function parse(sources, callback) {
  var results = {};

  var data = sources.reduce(function parser(memo, data) {
    // Try to repair some of the odd structures that are in the yaml files
    // before parsing it so we generate a uniform structure:

    // Normalize the Operating system versions:
    data = data.replace(/os_v([1-3])_replacement/gim, function replace(match, version) {
      return 'v'+ version +'_replacement';
    });

    // Make sure that we are able to parse the yaml string
    try { data = yaml.eval(data); }
    catch (e) {
      callback(e);
      callback = null;
      return memo;
    }

    // merge the data with the memo;
    Object.keys(data).forEach(function (key) {
      var results = data[key];
      memo[key] = memo[key] || [];

      for (var i = 0, l = results.length; i < l; i++) {
        memo[key].push(results[i]);
      }
    });

    return memo;
  }, {});

  [
      {
          resource: 'user_agent_parsers'
        , replacement: 'family_replacement'
        , name: 'browser'
      }
    , {
          resource: 'device_parsers'
        , replacement: 'device_replacement'
        , name: 'device'
      }
    , {
          resource: 'os_parsers'
        , replacement: 'os_replacement'
        , name: 'os'
      }
  ].forEach(function parsing(details) {
    results[details.resource] = results[details.resource] || [];

    var resources = data[details.resource]
      , name = details.resource.replace('_parsers', '')
      , resource
      , parser;

    for (var i = 0, l = resources.length; i < l; i++) {
      resource = resources[i];

      // We need to JSON stringify the data to properly add slashes escape other
      // kinds of crap in the RegularExpression. If we don't do thing we get
      // some illegal token warnings.
      parser = 'parser = Object.create(null);\n';
      parser += 'parser[0] = new RegExp('+ JSON.stringify(resource.regex) + ');\n';

      // Check if we have replacement for the parsed family name
      if (resource[details.replacement]) {
        parser += 'parser[1] = "'+ resource[details.replacement].replace('"', '\\"') +'";';
      } else {
        parser += 'parser[1] = 0;';
      }

      parser += '\n';

      if (resource.v1_replacement) {
        parser += 'parser[2] = "'+ resource.v1_replacement.replace('"', '\\"') +'";';
      } else {
        parser += 'parser[2] = 0;';
      }

      parser += '\n';

      if (resource.v2_replacement) {
        parser += 'parser[3] = "'+ resource.v2_replacement.replace('"', '\\"') +'";';
      } else {
        parser += 'parser[3] = 0;';
      }

      parser += '\n';

      if (resource.v3_replacement) {
        parser += 'parser[4] = "'+ resource.v3_replacement.replace('"', '\\"') +'";';
      } else {
        parser += 'parser[4] = 0;';
      }

      parser += '\n';
      parser += 'exports.'+ details.name +'['+ i +'] = parser;';
      results[details.resource].push(parser);
    }
  });

  // Generate a correct format
  exports.generate(results, callback);
};

exports.generate = function generate(results, callback) {
  var regexps  = [
      'var parser;'
    , 'exports.browser = Object.create(null);'
    , results.user_agent_parsers.join('\n')
    , 'exports.browser.length = '+ results.user_agent_parsers.length +';'

    , 'exports.device = Object.create(null);'
    , results.device_parsers.join('\n')
    , 'exports.device.length = '+ results.device_parsers.length +';'

    , 'exports.os = Object.create(null);'
    , results.os_parsers.join('\n')
    , 'exports.os.length = '+ results.os_parsers.length +';'
  ].join('\n\n');

  // Now that we have generated the structure for the RegExps export file we
  // need to validate that we created a JavaScript compatible file, if we would
  // write the file without checking it's content we could be breaking the
  // module.
  var sandbox = {
      exports: {} // Emulate a module context, so everything is attached here
  };

  // Crossing our fingers that it worked
  try { vm.runInNewContext(regexps, sandbox, 'validating.vm'); }
  catch (e) { return callback(e, null, regexps); }

  callback(undefined, sandbox.exports, regexps);
};

/**
 * The location of the ua-parser regexes yaml file.
 *
 * @type {String}
 * @api private
 */
exports.remote = 'https://raw.github.com/tobie/ua-parser/master/regexes.yaml';

/**
 * The location of our local regexes yaml file.
 *
 * @type {String}
 * @api private
 */
exports.local = path.resolve(__dirname, '..', 'static', 'user_agent.after.yaml');

/**
 * The the output location for the generated regexps file
 *
 * @type {String}
 * @api private
 */
exports.output = path.resolve(__dirname, '..', 'lib', 'regexps.js');
