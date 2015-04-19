"use strict";

var fs = require('fs'),
    Handlebars = require('./index'),
    basename = require('path').basename,
    uglify = require('uglify-js');

module.exports.cli = function(opts) {
  if (opts.version) {
    console.log(Handlebars.VERSION);
    return;
  }

  var template = [0];
  if (!opts.templates.length) {
    throw new Handlebars.Exception('Must define at least one template or directory.');
  }

  opts.templates.forEach(function(template) {
    try {
      fs.statSync(template);
    } catch (err) {
      throw new Handlebars.Exception('Unable to open template file "' + template + '"');
    }
  });

  if (opts.simple && opts.min) {
    throw new Handlebars.Exception('Unable to minimze simple output');
  }
  if (opts.simple && (opts.templates.length !== 1 || fs.statSync(opts.templates[0]).isDirectory())) {
    throw new Handlebars.Exception('Unable to output multiple templates in simple mode');
  }

  // Convert the known list into a hash
  var known = {};
  if (opts.known && !Array.isArray(opts.known)) {
    opts.known = [opts.known];
  }
  if (opts.known) {
    for (var i = 0, len = opts.known.length; i < len; i++) {
      known[opts.known[i]] = true;
    }
  }

  // Build file extension pattern
  var extension = opts.extension.replace(/[\\^$*+?.():=!|{}\-\[\]]/g, function(arg) { return '\\' + arg; });
  extension = new RegExp('\\.' + extension + '$');

  var output = [];
  if (!opts.simple) {
    if (opts.amd) {
      output.push('define([\'' + opts.handlebarPath + 'handlebars.runtime\'], function(Handlebars) {\n  Handlebars = Handlebars["default"];');
    } else if (opts.commonjs) {
      output.push('var Handlebars = require("' + opts.commonjs + '");');
    } else {
      output.push('(function() {\n');
    }
    output.push('  var template = Handlebars.template, templates = ');
    output.push(opts.namespace);
    output.push(' = ');
    output.push(opts.namespace);
    output.push(' || {};\n');
  }
  function processTemplate(template, root) {
    var path = template,
        stat = fs.statSync(path);
    if (stat.isDirectory()) {
      fs.readdirSync(template).map(function(file) {
        var path = template + '/' + file;

        if (extension.test(path) || fs.statSync(path).isDirectory()) {
          processTemplate(path, root || template);
        }
      });
    } else {
      var data = fs.readFileSync(path, 'utf8');

      if (opts.bom && data.indexOf('\uFEFF') === 0) {
        data = data.substring(1);
      }

      var options = {
        knownHelpers: known,
        knownHelpersOnly: opts.o
      };

      if (opts.data) {
        options.data = true;
      }

      // Clean the template name
      if (!root) {
        template = basename(template);
      } else if (template.indexOf(root) === 0) {
        template = template.substring(root.length+1);
      }
      template = template.replace(extension, '');

      if (opts.simple) {
        output.push(Handlebars.precompile(data, options) + '\n');
      } else if (opts.partial) {
        if(opts.amd && (opts.templates.length == 1 && !fs.statSync(opts.templates[0]).isDirectory())) {
          output.push('return ');
        }
        output.push('Handlebars.partials[\'' + template + '\'] = template(' + Handlebars.precompile(data, options) + ');\n');
      } else {
        if(opts.amd && (opts.templates.length == 1 && !fs.statSync(opts.templates[0]).isDirectory())) {
          output.push('return ');
        }
        output.push('templates[\'' + template + '\'] = template(' + Handlebars.precompile(data, options) + ');\n');
      }
    }
  }

  opts.templates.forEach(function(template) {
    processTemplate(template, opts.root);
  });

  // Output the content
  if (!opts.simple) {
    if (opts.amd) {
      if(opts.templates.length > 1 || (opts.templates.length == 1 && fs.statSync(opts.templates[0]).isDirectory())) {
        if(opts.partial){
          output.push('return Handlebars.partials;\n');
        } else {
          output.push('return templates;\n');
        }
      }
      output.push('});');
    } else if (!opts.commonjs) {
      output.push('})();');
    }
  }
  output = output.join('');

  if (opts.min) {
    output = uglify.minify(output, {fromString: true}).code;
  }

  if (opts.output) {
    fs.writeFileSync(opts.output, output, 'utf8');
  } else {
    console.log(output);
  }
};